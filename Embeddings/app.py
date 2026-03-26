import asyncio
import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

EMBED_MODEL_DIR = Path(os.getenv("EMBED_MODEL_DIR", os.getenv("MODEL_DIR", "/models/embedding")))
EMBED_CONFIG_PATH = Path(os.getenv("EMBED_SERVICE_CONFIG", EMBED_MODEL_DIR / "service_config.json"))
RERANK_MODEL_DIR = Path(os.getenv("RERANK_MODEL_DIR", "/models/reranker"))
RERANK_CONFIG_PATH = Path(os.getenv("RERANK_SERVICE_CONFIG", RERANK_MODEL_DIR / "service_config.json"))


EMBED_PRESETS = {
    "qwen3-embedding": {
        "pooling": "last_token",
        "normalize": True,
        "query_prefix": "Instruct: Given a search query, retrieve relevant passages.\nQuery: ",
        "document_prefix": "",
    },
    "bge-m3": {
        "pooling": "cls",
        "normalize": True,
        "query_prefix": "",
        "document_prefix": "",
    },
}


def infer_embed_preset(model_id: str) -> Optional[str]:
    lowered = model_id.lower()
    if "qwen3-embedding" in lowered:
        return "qwen3-embedding"
    if "bge-m3" in lowered:
        return "bge-m3"
    return None


def parse_api_keys() -> set[str]:
    raw = os.getenv("API_KEYS", "")
    return {x.strip() for x in raw.split(",") if x.strip()}


class EmbeddingRequest(BaseModel):
    input: Union[str, list[str], list[int], list[list[int]]]
    model: Optional[str] = None
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None
    input_type: Optional[Literal["query", "document"]] = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[list[float], str]


class UsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: UsageInfo


class RerankRequest(BaseModel):
    model: Optional[str] = None
    query: str
    documents: list[str] = Field(min_length=1)
    top_n: Optional[int] = None
    return_documents: bool = True
    max_length: Optional[int] = None
    user: Optional[str] = None


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: Optional[dict[str, str]] = None


class RerankResponse(BaseModel):
    object: Literal["list"] = "list"
    results: list[RerankResult]
    model: str
    usage: UsageInfo


class ORTBaseModel:
    def __init__(self, model_dir: Path, config_path: Path):
        if not model_dir.exists():
            raise RuntimeError(f"Model directory not found: {model_dir}")
        if not config_path.exists():
            raise RuntimeError(f"Config file not found: {config_path}")
        self.model_dir = model_dir
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.model_id = self.config["model_id"]
        self.max_length = int(self.config.get("max_length", 512))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.session = self._build_session(model_dir)
        self.input_names = {inp.name for inp in self.session.get_inputs()}

    def _build_session(self, model_dir: Path) -> ort.InferenceSession:
        onnx_files = sorted(model_dir.glob("**/*.onnx"))
        if not onnx_files:
            raise RuntimeError(f"No ONNX file found under {model_dir}")
        model_path = str(onnx_files[0])

        requested = os.getenv("ORT_PROVIDERS", "CUDAExecutionProvider,CPUExecutionProvider")
        available = set(ort.get_available_providers())
        providers: list[Any] = []
        for item in [x.strip() for x in requested.split(",") if x.strip()]:
            if item in available:
                if item == "CUDAExecutionProvider":
                    providers.append((
                        "CUDAExecutionProvider",
                        {
                            "device_id": int(os.getenv("CUDA_DEVICE_ID", "0")),
                            "arena_extend_strategy": "kSameAsRequested",
                            "gpu_mem_limit": str(int(float(os.getenv("ORT_GPU_MEM_LIMIT_GB", "14")) * 1024**3)),
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    ))
                else:
                    providers.append(item)
        if not providers:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        intra = int(os.getenv("ORT_INTRA_OP_THREADS", "0"))
        inter = int(os.getenv("ORT_INTER_OP_THREADS", "0"))
        if intra > 0:
            sess_options.intra_op_num_threads = intra
        if inter > 0:
            sess_options.inter_op_num_threads = inter
        return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

    def _select_inputs(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v for k, v in feeds.items() if k in self.input_names}


class EmbeddingModel(ORTBaseModel):
    def __init__(self, model_dir: Path, config_path: Path):
        super().__init__(model_dir, config_path)
        preset_name = self.config.get("preset") or infer_embed_preset(self.model_id)
        preset = EMBED_PRESETS.get(preset_name, {})
        self.preset = preset_name
        self.pooling = self.config.get("pooling", preset.get("pooling", "mean"))
        self.normalize = bool(self.config.get("normalize", preset.get("normalize", True)))
        self.query_prefix = self.config.get("query_prefix", preset.get("query_prefix", ""))
        self.document_prefix = self.config.get("document_prefix", preset.get("document_prefix", ""))

        # 缓存 ONNX 模型实际需要的输入名字
        self.session_input_names = {x.name for x in self.session.get_inputs()}

        print(f"[EmbeddingModel] model_id={self.model_id}")
        print(f"[EmbeddingModel] session inputs={sorted(self.session_input_names)}")
        print(f"[EmbeddingModel] pooling={self.pooling}, normalize={self.normalize}, preset={self.preset}")

    def _prepare_text(self, text: str, input_type: Optional[str]) -> str:
        if input_type == "query" and self.query_prefix:
            return self.query_prefix + text
        if input_type == "document" and self.document_prefix:
            return self.document_prefix + text
        return text

    def _build_position_ids(self, attention_mask: np.ndarray) -> np.ndarray:
        """
        为需要 position_ids 的模型构造位置编码输入。
        对 padding 场景更稳妥：
        attention_mask = [0, 0, 1, 1, 1] -> position_ids = [0, 0, 0, 1, 2]
        attention_mask = [1, 1, 1, 0, 0] -> position_ids = [0, 1, 2, 0, 0]
        """
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids = np.clip(position_ids, 0, None).astype(np.int64)
        position_ids = position_ids * attention_mask
        return position_ids

    def _tokenize(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        feeds: dict[str, np.ndarray] = {}

        if "input_ids" in self.session_input_names:
            feeds["input_ids"] = input_ids

        if "attention_mask" in self.session_input_names:
            feeds["attention_mask"] = attention_mask

        if "position_ids" in self.session_input_names:
            feeds["position_ids"] = self._build_position_ids(attention_mask)

        if "token_type_ids" in self.session_input_names:
            if "token_type_ids" in enc:
                feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
            else:
                feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        return feeds

    def _pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask.astype(np.float32)[..., None]

        if self.pooling == "cls":
            sent = token_embeddings[:, 0, :]

        elif self.pooling == "last_token":
            lengths = attention_mask.sum(axis=1) - 1
            lengths = np.clip(lengths, 0, token_embeddings.shape[1] - 1)
            sent = token_embeddings[np.arange(token_embeddings.shape[0]), lengths]

        else:  # mean
            masked = token_embeddings * mask
            summed = masked.sum(axis=1)
            denom = np.clip(mask.sum(axis=1), 1e-9, None)
            sent = summed / denom

        if self.normalize:
            norms = np.linalg.norm(sent, axis=1, keepdims=True)
            sent = sent / np.clip(norms, 1e-12, None)

        return sent.astype(np.float32)

    def embed(
        self,
        inputs: list[str],
        input_type: Optional[str],
        dimensions: Optional[int],
    ) -> tuple[np.ndarray, int]:
        texts = [self._prepare_text(x, input_type) for x in inputs]
        feeds = self._tokenize(texts)

        outputs = self.session.run(None, self._select_inputs(feeds))
        #token_embeddings = outputs[0]

        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(None, self._select_inputs(feeds))

        if len(outputs) == 0:
            raise HTTPException(status_code=500, detail="ONNX model returned no outputs")

        token_embeddings = outputs[0]

        if "attention_mask" not in feeds:
            raise HTTPException(
                status_code=500,
                detail="ONNX model inputs do not include attention_mask; current pooling implementation requires it.",
            )

        sentence_embeddings = self._pool(token_embeddings, feeds["attention_mask"])

        if dimensions is not None:
            if dimensions <= 0 or dimensions > sentence_embeddings.shape[1]:
                raise HTTPException(
                    status_code=400,
                    detail=f"dimensions must be between 1 and {sentence_embeddings.shape[1]}",
                )
            sentence_embeddings = sentence_embeddings[:, :dimensions]

        token_count = int(feeds["attention_mask"].sum())
        return sentence_embeddings, token_count


class RerankerModel(ORTBaseModel):
    def __init__(self, model_dir: Path, config_path: Path):
        super().__init__(model_dir, config_path)
        self.normalize_scores = bool(self.config.get("normalize_scores", False))
        self.query_prefix = self.config.get("query_prefix", "")
        self.document_prefix = self.config.get("document_prefix", "")

        # 缓存 ONNX 模型实际输入
        self.session_input_names = {x.name for x in self.session.get_inputs()}

        print(f"[RerankerModel] model_id={self.model_id}")
        print(f"[RerankerModel] session inputs={sorted(self.session_input_names)}")
        print(f"[RerankerModel] normalize_scores={self.normalize_scores}")

    def _build_position_ids(self, attention_mask: np.ndarray) -> np.ndarray:
        position_ids = np.cumsum(attention_mask, axis=1) - 1
        position_ids = np.clip(position_ids, 0, None).astype(np.int64)
        position_ids = position_ids * attention_mask
        return position_ids

    def _tokenize_pairs(
        self,
        query: str,
        documents: list[str],
        max_length: Optional[int],
    ) -> dict[str, np.ndarray]:
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        docs = [f"{self.document_prefix}{d}" if self.document_prefix else d for d in documents]

        enc = self.tokenizer(
            [query_text] * len(docs),
            docs,
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length,
            return_tensors="np",
        )

        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)

        feeds: dict[str, np.ndarray] = {}

        if "input_ids" in self.session_input_names:
            feeds["input_ids"] = input_ids

        if "attention_mask" in self.session_input_names:
            feeds["attention_mask"] = attention_mask

        if "position_ids" in self.session_input_names:
            feeds["position_ids"] = self._build_position_ids(attention_mask)

        if "token_type_ids" in self.session_input_names:
            if "token_type_ids" in enc:
                feeds["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
            else:
                feeds["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

        return feeds

    def rerank(
        self,
        query: str,
        documents: list[str],
        max_length: Optional[int],
    ) -> tuple[list[float], int]:
        feeds = self._tokenize_pairs(query, documents, max_length)

        outputs = self.session.run(None, self._select_inputs(feeds))
        if len(outputs) == 0:
            raise HTTPException(status_code=500, detail="ONNX reranker returned no outputs")

        logits = outputs[0]

        if logits.ndim == 2:
            if logits.shape[1] == 1:
                scores = logits[:, 0]
            else:
                scores = logits[:, -1]
        else:
            scores = logits.reshape(-1)

        scores = scores.astype(np.float32)

        if self.normalize_scores:
            scores = 1.0 / (1.0 + np.exp(-scores))

        if "attention_mask" not in feeds:
            raise HTTPException(
                status_code=500,
                detail="ONNX reranker inputs do not include attention_mask; token counting requires it.",
            )

        token_count = int(feeds["attention_mask"].sum())
        return scores.astype(float).tolist(), token_count


class EmbeddingBatchItem:
    def __init__(self, texts: list[str], input_type: Optional[str], dimensions: Optional[int]):
        self.texts = texts
        self.input_type = input_type
        self.dimensions = dimensions
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()


class RerankBatchItem:
    def __init__(self, query: str, docs: list[str], max_length: Optional[int]):
        self.query = query
        self.docs = docs
        self.max_length = max_length
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()


class Batcher:
    def __init__(self, embed_model: EmbeddingModel, reranker_model: Optional[RerankerModel]):
        self.embed_model = embed_model
        self.reranker_model = reranker_model
        self.embed_queue: asyncio.Queue[EmbeddingBatchItem] = asyncio.Queue()
        self.rerank_queue: asyncio.Queue[RerankBatchItem] = asyncio.Queue()
        self.embed_max_items = int(os.getenv("EMBED_MAX_BATCH_TEXTS", "32"))
        self.embed_max_wait_ms = int(os.getenv("EMBED_BATCH_TIMEOUT_MS", "8"))
        self.rerank_max_pairs = int(os.getenv("RERANK_MAX_BATCH_PAIRS", "32"))
        self.rerank_max_wait_ms = int(os.getenv("RERANK_BATCH_TIMEOUT_MS", "8"))
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(self._embed_loop(), name="embed-batcher"))
        if self.reranker_model is not None:
            self._tasks.append(asyncio.create_task(self._rerank_loop(), name="rerank-batcher"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def enqueue_embed(self, item: EmbeddingBatchItem) -> tuple[np.ndarray, int]:
        await self.embed_queue.put(item)
        return await item.future

    async def enqueue_rerank(self, item: RerankBatchItem) -> tuple[list[float], int]:
        if self.reranker_model is None:
            raise HTTPException(status_code=404, detail="reranker model is not configured")
        await self.rerank_queue.put(item)
        return await item.future

    async def _gather_items(self, queue: asyncio.Queue, max_units: int, timeout_ms: int, unit_counter) -> list:
        first = await queue.get()
        items = [first]
        units = unit_counter(first)
        deadline = asyncio.get_running_loop().time() + timeout_ms / 1000.0
        while units < max_units:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(queue.get(), timeout=remaining)
                next_units = unit_counter(nxt)
                if units + next_units > max_units and items:
                    queue.put_nowait(nxt)
                    break
                items.append(nxt)
                units += next_units
            except asyncio.TimeoutError:
                break
        return items

    async def _embed_loop(self) -> None:
        while True:
            items = await self._gather_items(self.embed_queue, self.embed_max_items, self.embed_max_wait_ms, lambda x: len(x.texts))
            groups: dict[tuple[Optional[str], Optional[int]], list[EmbeddingBatchItem]] = {}
            for item in items:
                groups.setdefault((item.input_type, item.dimensions), []).append(item)
            for (input_type, dimensions), chunk in groups.items():
                flat_texts: list[str] = []
                lengths: list[int] = []
                for item in chunk:
                    flat_texts.extend(item.texts)
                    lengths.append(len(item.texts))
                try:
                    vectors, tokens = self.embed_model.embed(flat_texts, input_type, dimensions)
                    offset = 0
                    for item, size in zip(chunk, lengths):
                        item.future.set_result((vectors[offset:offset + size], tokens))
                        offset += size
                except Exception as e:
                    for item in chunk:
                        if not item.future.done():
                            item.future.set_exception(e)

    async def _rerank_loop(self) -> None:
        while True:
            items = await self._gather_items(self.rerank_queue, self.rerank_max_pairs, self.rerank_max_wait_ms, lambda x: len(x.docs))
            if self.reranker_model is None:
                for item in items:
                    item.future.set_exception(HTTPException(status_code=404, detail="reranker model is not configured"))
                continue
            for item in items:
                try:
                    result = self.reranker_model.rerank(item.query, item.docs, item.max_length)
                    item.future.set_result(result)
                except Exception as e:
                    if not item.future.done():
                        item.future.set_exception(e)


class ServiceState:
    def __init__(self):
        self.api_keys = parse_api_keys()
        self.embed_model = EmbeddingModel(EMBED_MODEL_DIR, EMBED_CONFIG_PATH)
        self.reranker_model: Optional[RerankerModel] = None
        if RERANK_MODEL_DIR.exists() and RERANK_CONFIG_PATH.exists():
            self.reranker_model = RerankerModel(RERANK_MODEL_DIR, RERANK_CONFIG_PATH)
        self.batcher = Batcher(self.embed_model, self.reranker_model)


state: Optional[ServiceState] = None
app = FastAPI(title="OpenAI-Compatible Embeddings and Rerank API (ONNX Runtime)")


async def require_api_key(authorization: Optional[str], x_api_key: Optional[str]) -> None:
    assert state is not None
    if not state.api_keys:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token not in state.api_keys:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in {"/", "/healthz", "/docs", "/openapi.json", "/redoc"} and os.getenv("AUTH_ON_HEALTH", "false").lower() != "true":
        return await call_next(request)
    await require_api_key(request.headers.get("authorization"), request.headers.get("x-api-key"))
    return await call_next(request)


@app.on_event("startup")
async def startup_event() -> None:
    global state
    state = ServiceState()
    await state.batcher.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    assert state is not None
    await state.batcher.stop()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "onnx-embeddings-rerank",
        "docs": "/docs",
        "health": "/healthz",
        "endpoints": ["/v1/models", "/v1/embeddings", "/v1/rerank"],
        "request_id_example": str(uuid.uuid4()),
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    assert state is not None
    return {
        "ok": True,
        "embedding_model": state.embed_model.model_id,
        "reranker_model": state.reranker_model.model_id if state.reranker_model else None,
        "providers": state.embed_model.session.get_providers(),
        "embedding_model_dir": str(EMBED_MODEL_DIR),
        "reranker_model_dir": str(RERANK_MODEL_DIR) if state.reranker_model else None,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    assert state is not None
    now = int(time.time())
    data = [
        {
            "id": state.embed_model.model_id,
            "object": "model",
            "created": now,
            "owned_by": "local-onnx-runtime",
            "capabilities": ["embeddings"],
        }
    ]
    if state.reranker_model is not None:
        data.append(
            {
                "id": state.reranker_model.model_id,
                "object": "model",
                "created": now,
                "owned_by": "local-onnx-runtime",
                "capabilities": ["rerank"],
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    req: EmbeddingRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> EmbeddingResponse:
    await require_api_key(authorization, x_api_key)
    assert state is not None
    if req.model and req.model != state.embed_model.model_id:
        raise HTTPException(status_code=400, detail=f"loaded embedding model is '{state.embed_model.model_id}', not '{req.model}'")
    if isinstance(req.input, str):
        texts = [req.input]
    elif isinstance(req.input, list) and (not req.input or isinstance(req.input[0], str)):
        texts = req.input
    else:
        raise HTTPException(status_code=400, detail="this wrapper currently supports string input or list[str]")
    vectors, tokens = await state.batcher.enqueue_embed(EmbeddingBatchItem(texts, req.input_type, req.dimensions))
    items: list[EmbeddingItem] = []
    for i, vec in enumerate(vectors):
        if req.encoding_format == "base64":
            payload: Union[list[float], str] = base64.b64encode(vec.astype(np.float32).tobytes()).decode("ascii")
        else:
            payload = vec.astype(float).tolist()
        items.append(EmbeddingItem(index=i, embedding=payload))
    return EmbeddingResponse(
        data=items,
        model=state.embed_model.model_id,
        usage=UsageInfo(prompt_tokens=tokens, total_tokens=tokens),
    )


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> RerankResponse:
    await require_api_key(authorization, x_api_key)
    assert state is not None
    if state.reranker_model is None:
        raise HTTPException(status_code=404, detail="reranker model is not configured")
    if req.model and req.model != state.reranker_model.model_id:
        raise HTTPException(status_code=400, detail=f"loaded reranker model is '{state.reranker_model.model_id}', not '{req.model}'")
    scores, tokens = await state.batcher.enqueue_rerank(RerankBatchItem(req.query, req.documents, req.max_length))
    ranked = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)
    if req.top_n is not None:
        ranked = ranked[:req.top_n]
    results: list[RerankResult] = []
    for idx, score in ranked:
        doc_payload = {"text": req.documents[idx]} if req.return_documents else None
        results.append(RerankResult(index=idx, relevance_score=float(score), document=doc_payload))
    return RerankResponse(
        results=results,
        model=state.reranker_model.model_id,
        usage=UsageInfo(prompt_tokens=tokens, total_tokens=tokens),
    )

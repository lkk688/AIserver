import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a sentence-transformers embedding model to ONNX.")
    parser.add_argument("--model-id", required=True, help="HF model id, e.g. BAAI/bge-m3 or Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--output-dir", required=True, help="Directory to write ONNX files into")
    parser.add_argument("--pooling", default="mean", choices=["mean", "cls", "last_token"], help="Pooling to apply in the wrapper")
    parser.add_argument("--normalize", action="store_true", help="L2 normalize embeddings in the wrapper")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length")
    parser.add_argument("--query-prefix", default="", help="Prefix added when input_type=query")
    parser.add_argument("--document-prefix", default="", help="Prefix added when input_type=document")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model_id, backend="onnx")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model.save(str(out))
    tokenizer.save_pretrained(str(out))

    config = {
        "model_id": args.model_id,
        "task": "embedding",
        "max_length": args.max_length,
        "pooling": args.pooling,
        "normalize": args.normalize,
        "query_prefix": args.query_prefix,
        "document_prefix": args.document_prefix,
        "default_encoding_format": "float",
    }
    with open(out / "service_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Exported embedding model and tokenizer to {out}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a reranker / sequence-classification model to ONNX.")
    parser.add_argument("--model-id", required=True, help="HF model id, e.g. Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--output-dir", required=True, help="Directory to write ONNX files into")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer truncation length")
    parser.add_argument("--normalize-scores", action="store_true", help="Apply sigmoid to logits in the wrapper")
    parser.add_argument("--query-prefix", default="", help="Optional prefix for query text")
    parser.add_argument("--document-prefix", default="", help="Optional prefix for document text")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(args.model_id, export=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    config = {
        "model_id": args.model_id,
        "task": "rerank",
        "max_length": args.max_length,
        "normalize_scores": args.normalize_scores,
        "query_prefix": args.query_prefix,
        "document_prefix": args.document_prefix,
    }
    with open(out / "service_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Exported reranker model and tokenizer to {out}")


if __name__ == "__main__":
    main()

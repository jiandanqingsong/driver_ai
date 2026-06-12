from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver_distraction.deploy.acl_infer import AscendOMClassifier, preprocess_image


DEFAULT_MODEL = "deploy/models/mobilenet_v3_large_demo_finetune_driver_distraction.om"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-image inference with an Ascend OM model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to the OM model.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument(
        "--output-dtype",
        choices=("auto", "float32", "float16"),
        default="auto",
        help="Use auto for the current 1x10 FP32 output model.",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=0,
        help="Run repeated inference and print latency statistics when greater than zero.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

    with AscendOMClassifier(
        model_path=model_path,
        device_id=args.device_id,
        output_dtype=args.output_dtype,
    ) as classifier:
        predictions = classifier.predict(
            image_path,
            top_k=args.top_k,
            input_size=args.input_size,
            resize_size=args.resize_size,
        )

        benchmark = None
        if args.benchmark_runs > 0:
            tensor = preprocess_image(
                image_path,
                input_size=args.input_size,
                resize_size=args.resize_size,
            )
            benchmark = classifier.benchmark(
                tensor,
                warmup_runs=args.warmup_runs,
                measured_runs=args.benchmark_runs,
            )

    result = {
        "model": str(model_path),
        "image": str(image_path),
        "device_id": args.device_id,
        "prediction": {
            "index": predictions[0].index,
            "label": predictions[0].label,
            "label_zh": predictions[0].label_zh,
            "probability": predictions[0].probability,
        },
        "top_k": [
            {
                "index": item.index,
                "label": item.label,
                "label_zh": item.label_zh,
                "probability": item.probability,
            }
            for item in predictions
        ],
        "benchmark": benchmark,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    best = predictions[0]
    print(f"模型: {model_path}")
    print(f"图像: {image_path}")
    print(f"预测结果: C{best.index} {best.label_zh} ({best.label})")
    print(f"置信度: {best.probability:.2%}")
    print("\nTop-K:")
    for rank, item in enumerate(predictions, start=1):
        print(
            f"  {rank}. C{item.index} {item.label_zh:<10} "
            f"{item.probability:>7.2%}  {item.label}"
        )

    if benchmark is not None:
        print("\n性能:")
        print(
            f"  平均 {benchmark['mean_ms']:.2f} ms, "
            f"P50 {benchmark['p50_ms']:.2f} ms, "
            f"P95 {benchmark['p95_ms']:.2f} ms, "
            f"约 {benchmark['fps']:.2f} FPS"
        )


if __name__ == "__main__":
    main()

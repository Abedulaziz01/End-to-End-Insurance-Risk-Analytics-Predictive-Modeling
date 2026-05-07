from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insurance_risk.data import DEFAULT_TARGET, REAL_DATA_PATH, load_dataset
from insurance_risk.modeling import DEFAULT_METRICS_PATH, DEFAULT_MODEL_PATH, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the insurance premium model.")
    parser.add_argument("--data", type=Path, default=REAL_DATA_PATH, help="Path to the cleaned CSV file.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column to predict.")
    parser.add_argument("--demo-rows", type=int, default=5000, help="Rows to generate in demo mode.")
    parser.add_argument("--max-rows", type=int, default=30000, help="Max rows to train on when real data exists.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Path to save the model artifact.")
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH, help="Path to save metrics JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, mode = load_dataset(
        path=args.data,
        target=args.target,
        demo_rows=args.demo_rows,
        max_rows=args.max_rows,
    )
    _, metrics, feature_columns = train_model(df, target=args.target, model_path=args.model_out)

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_mode": mode,
        "rows_used": int(len(df)),
        "target": args.target,
        "feature_count": len(feature_columns),
        "metrics": metrics,
    }
    args.metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

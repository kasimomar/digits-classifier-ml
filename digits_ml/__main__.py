"""Run `python -m digits_ml --help` for CLI usage."""
import argparse
import json
from pathlib import Path
import joblib
from .model import train, validate_pixels, save_confusion_plot


def main():
    parser = argparse.ArgumentParser(description="Digits benchmark and validated batch inference")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    fit.add_argument("--seed", type=int, default=42)
    predict = sub.add_parser("predict", help="Load only trusted, locally trained joblib files")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            report, _, matrix = train(args.output_dir, args.seed)
            save_confusion_plot(matrix, args.output_dir / "confusion-matrix.png")
            print(json.dumps({key: report[key] for key in ["selected_model", "baseline_test", "selected_test"]}, indent=2))
        else:
            pixels = validate_pixels(json.loads(args.input.read_text()))
            model = joblib.load(args.model)
            print(json.dumps({"predictions": model.predict(pixels).astype(int).tolist()}))
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

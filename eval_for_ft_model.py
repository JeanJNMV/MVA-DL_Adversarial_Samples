import argparse
import json
from pathlib import Path
from ultralytics import YOLO

from constant import DEFAULT_EPS_LIST


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run built-in YOLO validation on adversarial datasets."
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=DEFAULT_EPS_LIST,
        help="List of epsilon values to evaluate (default: all values in DEFAULT_EPS_LIST).",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="models/yolo11n_000.pt",
        help="Path to the model .pt file. MUST BE FINE-TUNED ON VOC ! (default: models/yolo11n_000.pt)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If None, saves to results/<model_name>/results.json",
    )

    return parser.parse_args()


def run_eval():
    args = parse_arguments()

    model_path = args.model_name
    model_name_stem = Path(model_path).stem
    root = Path(".")

    eps_list = args.epsilon

    results_dir = Path(f"results/{model_name_stem}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        json_path = Path(args.output)
    else:
        json_path = results_dir / "results.json"

    if json_path.exists():
        with open(json_path, "r") as f:
            try:
                results_dict = json.load(f)
            except json.JSONDecodeError:
                results_dict = {}
    else:
        results_dict = {}

    device = "cpu"

    print(f"\nLoading model '{model_path}' on {device.upper()}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    for eps in eps_list:
        print(f"\n=== Evaluating epsilon = {eps:.2f} ===")

        yaml_path = root / f"adv_VOC_YOLO_eps_{eps:.2f}/dataset.yaml"

        if not yaml_path.exists():
            print(f"Skipping epsilon={eps:.2f}: {yaml_path} not found.")
            continue

        try:
            # Run built-in validation
            val_project = f"runs/val/{model_name_stem}"
            val_name = f"eps_{eps:.2f}"

            results = model.val(
                data=str(yaml_path),
                batch=4,
                device=device,
                imgsz=640,
                verbose=True,
                plots=True,
                project=val_project,
                name=val_name,
                exist_ok=True,  # Overwrite existing runs folder if needed
            )

            # Extract mAP@50
            map50 = float(results.box.map50)

            print(f"mAP@50 (eps={eps:.2f}) = {map50:.4f}")

            # Update dictionary
            results_dict[f"{eps:.2f}"] = map50

        except Exception as e:
            print(f"Error evaluating epsilon={eps:.2f}: {e}")

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print("\nSaved results to:", json_path)


if __name__ == "__main__":
    run_eval()

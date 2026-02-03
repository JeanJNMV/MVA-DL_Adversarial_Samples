import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from constant import DEFAULT_EPS_LIST, COCO_TO_VOC
from utils import evaluate_map


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model robustness against adversarial attacks."
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
        default="models/yolo11n.pt",
        help="Path to the model .pt file (default: models/yolo11n.pt)",
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

    eps_list = args.epsilon

    root = Path(".")
    results_dir = Path(f"results/{model_name_stem}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        json_path = Path(args.output)
    else:
        json_path = results_dir / "results.json"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"\nLoading model '{model_path}' on {device.upper()}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    n_classes = len(model.names)
    if n_classes == 80:
        remap = True
        print("Detected COCO model (80 classes), using COCO to VOC remap")
    elif n_classes == 20:
        remap = False
        print("Detected VOC model (20 classes), no remap needed")
    else:
        print(
            f"Warning: Model has {n_classes} classes. Assuming direct mapping if consistent with VOC."
        )
        remap = False

    if json_path.exists():
        with open(json_path, "r") as f:
            try:
                results_dict = json.load(f)
            except json.JSONDecodeError:
                results_dict = {}
    else:
        results_dict = {}

    for eps in eps_list:
        print(f"\n=== Evaluating epsilon = {eps:.2f} ===")

        images_folder = root / f"adv_VOC_YOLO_eps_{eps:.2f}/images/val"
        labels_folder = root / f"adv_VOC_YOLO_eps_{eps:.2f}/labels/val"

        if not images_folder.exists():
            print(f"Skipping epsilon={eps:.2f}: Folder {images_folder} not found.")
            continue

        preds_dir = Path(f"predictions/{model_name_stem}_eps_{eps:.2f}")
        preds_dir.mkdir(parents=True, exist_ok=True)

        # Inference
        img_files = list(images_folder.glob("*.*"))

        for img_path in tqdm(img_files, desc=f"Inference epsilon {eps:.2f}"):
            # Run prediction
            result = model.predict(img_path, verbose=False, device=device)[0]

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            H, W = result.orig_shape

            out_file = preds_dir / f"{img_path.stem}.txt"
            with open(out_file, "w") as f:
                for box, score, raw_cid in zip(boxes, scores, classes):
                    if remap:
                        coco_id = int(raw_cid)
                        if coco_id not in COCO_TO_VOC:
                            continue
                        cid = COCO_TO_VOC[coco_id]
                    else:
                        cid = int(raw_cid)

                    x1, y1, x2, y2 = box
                    # Normalize coordinates
                    xc = (x1 + x2) / 2 / W
                    yc = (y1 + y2) / 2 / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H

                    f.write(f"{cid} {xc} {yc} {w} {h} {score}\n")

        print("Computing mAP@50...")
        mAP = evaluate_map(labels_folder, preds_dir)
        print(f"mAP@50 (eps={eps:.2f}) = {mAP:.4f}")

        # Update dictionary
        results_dict[f"{eps:.2f}"] = float(mAP)

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print("\nSaved results to:", json_path)


if __name__ == "__main__":
    run_eval()

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

from constant import DEFAULT_EPS_LIST, COCO_TO_VOC_SSD
from utils import evaluate_map

DEFAULT_MODEL_NAME = "ssdlite_mobilenet_v3"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate SSDLite Robustness")
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
        default=DEFAULT_MODEL_NAME,
        help=f"Model name for output folder (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If None, saves to results/<model_name>/results.json",
    )
    return parser.parse_args()


def load_ssdlite(device):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights="DEFAULT"
    ).to(device)
    model.eval()
    return model


def run_eval():
    args = parse_arguments()

    root = Path(".")

    eps_list = args.epsilon
    model_name = args.model_name

    results_dir = Path(f"results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output is not None:
        json_path = Path(args.output)
    else:
        json_path = results_dir / "results.json"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nLoading SSD-Lite on {device}...")
    model = load_ssdlite(device)

    SCORE_THRESHOLD = 0.1

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
            print(f"Skipping epsilon={eps:.2f}: Folder not found at {images_folder}")
            continue

        preds_dir = Path(f"predictions/{model_name}_eps_{eps:.2f}")
        preds_dir.mkdir(parents=True, exist_ok=True)

        img_files = list(images_folder.glob("*.*"))

        for img_path in tqdm(img_files, desc=f"Inference epsilon {eps:.2f}"):
            try:
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
                tensor = F.to_tensor(img).to(device)

                with torch.no_grad():
                    output = model([tensor])[0]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                classes = output["labels"].cpu().numpy()

                out_file = preds_dir / f"{img_path.stem}.txt"
                with open(out_file, "w") as f:
                    for box, score, raw_cid in zip(boxes, scores, classes):
                        if raw_cid == 0:  # skip background
                            continue
                        if score < SCORE_THRESHOLD:  # ignore low-confidence
                            continue
                        if int(raw_cid) not in COCO_TO_VOC_SSD:
                            continue
                        cid = COCO_TO_VOC_SSD[int(raw_cid)]

                        x1, y1, x2, y2 = box
                        xc = (x1 + x2) / 2 / W
                        yc = (y1 + y2) / 2 / H
                        w = (x2 - x1) / W
                        h = (y2 - y1) / H

                        f.write(f"{cid} {xc} {yc} {w} {h} {score}\n")
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print("Computing mAP@50...")
        mAP = evaluate_map(labels_folder, preds_dir)
        print(f"mAP@50 (eps={eps:.2f}) = {mAP:.4f}")

        # Update results
        results_dict[f"{eps:.2f}"] = float(mAP)

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)
    print("\nSaved results to:", json_path)


if __name__ == "__main__":
    run_eval()

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image

DEFAULT_EPS_LIST = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.17, 0.20]

MODEL_NAME = "ssdlite_mobilenet_v3"
ROOT = Path(".")  # root folder where adv_VOC_YOLO_eps_xx folders are
RESULTS_DIR = Path(f"results_curves/{MODEL_NAME}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate SSDLite Robustness")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Specific epsilon to evaluate. If None, runs all default values.",
    )
    return parser.parse_args()


COCO_TO_VOC = {
    5: 0,  # airplane    -> aeroplane
    2: 1,  # bicycle     -> bicycle
    16: 2,  # bird        -> bird
    9: 3,  # boat        -> boat
    44: 4,  # bottle      -> bottle
    6: 5,  # bus         -> bus
    3: 6,  # car         -> car
    17: 7,  # cat         -> cat
    62: 8,  # chair       -> chair
    21: 9,  # cow         -> cow
    67: 10,  # dining table -> diningtable
    18: 11,  # dog         -> dog
    19: 12,  # horse       -> horse
    4: 13,  # motorcycle  -> motorbike
    1: 14,  # person      -> person
    64: 15,  # potted plant -> pottedplant
    20: 16,  # sheep       -> sheep
    63: 17,  # couch       -> sofa
    7: 18,  # train       -> train
    72: 19,  # tv          -> tvmonitor
}

VOC_CLASS_COUNT = 20


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / (area1 + area2 - interArea + 1e-6)


def voc_ap(rec, prec):
    mprec = np.concatenate(([0.0], prec, [0.0]))
    mrec = np.concatenate(([0.0], rec, [1.0]))
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mprec[idx + 1])


def evaluate_map(gt_folder, pred_folder, iou_thresh=0.5, num_classes=VOC_CLASS_COUNT):
    aps = []
    gt_files = {f.stem: f for f in Path(gt_folder).glob("*.txt")}
    pred_files = {f.stem: f for f in Path(pred_folder).glob("*.txt")}

    for cls_id in range(num_classes):
        scores, matches = [], []
        npos = 0

        for stem, gt_file in gt_files.items():
            pred_file = pred_files.get(stem, None)

            # GT boxes
            gt_boxes = []
            with open(gt_file) as f:
                for line in f:
                    parts = line.split()
                    cid = int(parts[0])
                    if cid != cls_id:
                        continue
                    xc, yc, w, h = map(float, parts[1:])
                    x1 = xc - w / 2
                    y1 = yc - h / 2
                    x2 = xc + w / 2
                    y2 = yc + h / 2
                    gt_boxes.append([x1, y1, x2, y2])
            npos += len(gt_boxes)

            # Predictions
            pred_boxes = []
            if pred_file and pred_file.exists():
                with open(pred_file) as f:
                    for line in f:
                        parts = line.split()
                        cid = int(parts[0])
                        if cid != cls_id:
                            continue
                        xc, yc, w, h, score = map(float, parts[1:])
                        x1 = xc - w / 2
                        y1 = yc - h / 2
                        x2 = xc + w / 2
                        y2 = yc + h / 2
                        pred_boxes.append([x1, y1, x2, y2, score])

            pred_boxes.sort(key=lambda x: x[4], reverse=True)
            used = set()
            for pb in pred_boxes:
                best_iou = 0
                best_j = -1
                for j, gb in enumerate(gt_boxes):
                    if j in used:
                        continue
                    iou = compute_iou(pb[:4], gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                scores.append(pb[4])
                if best_iou >= iou_thresh and best_j >= 0:
                    matches.append(1)
                    used.add(best_j)
                else:
                    matches.append(0)

        if len(scores) == 0:
            aps.append(0.0)
            continue

        scores = np.array(scores)
        matches = np.array(matches)
        order = np.argsort(-scores)
        matches = matches[order]
        tp = np.cumsum(matches)
        fp = np.cumsum(1 - matches)
        rec = tp / (npos + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        aps.append(voc_ap(rec, prec))

    return np.mean(aps)


def load_ssdlite(device):
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
        weights="DEFAULT"
    ).to(device)
    model.eval()
    return model


def run_eval():
    args = parse_arguments()

    if args.epsilon is not None:
        eps_list = [args.epsilon]
    else:
        eps_list = DEFAULT_EPS_LIST

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nLoading SSD-Lite on {device}...")
    model = load_ssdlite(device)

    SCORE_THRESHOLD = 0.1

    json_path = RESULTS_DIR / "results.json"
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

        images_folder = ROOT / f"adv_VOC_YOLO_eps_{eps:.2f}/images/val"
        labels_folder = ROOT / f"adv_VOC_YOLO_eps_{eps:.2f}/labels/val"

        if not images_folder.exists():
            print(f"Skipping epsilon={eps:.2f}: Folder not found at {images_folder}")
            continue

        preds_dir = Path(f"predictions/{MODEL_NAME}_eps_{eps:.2f}")
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
                        if int(raw_cid) not in COCO_TO_VOC:
                            continue
                        cid = COCO_TO_VOC[int(raw_cid)]

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

import numpy as np
from pathlib import Path

from constant import VOC_CLASS_COUNT


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

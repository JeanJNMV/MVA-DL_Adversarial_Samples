import os
import xml.etree.ElementTree as ET
import shutil
import argparse

# VOC classes
CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def voc_to_yolo_bbox(bbox, img_w, img_h):
    xmin, ymin, xmax, ymax = bbox
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x_center, y_center, w, h


def convert_annotation(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in CLASSES:
            continue
        cls_id = CLASSES.index(cls_name)

        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        ymin = float(xmlbox.find("ymin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymax = float(xmlbox.find("ymax").text)

        bb = voc_to_yolo_bbox((xmin, ymin, xmax, ymax), img_w, img_h)
        line = "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(cls_id, *bb)
        lines.append(line)

    with open(output_txt, "w") as f:
        f.write("\n".join(lines))


def process_split(voc_root, split, out_img_dir, out_lbl_dir):
    split_file = os.path.join(voc_root, "ImageSets/Main", split + ".txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    ann_dir = os.path.join(voc_root, "Annotations")

    if not os.path.exists(split_file):
        print(f"Warning: Split file not found at {split_file}. Skipping.")
        return

    with open(split_file) as f:
        img_ids = [x.strip() for x in f.readlines()]

    for img_id in img_ids:
        jpg_src = os.path.join(img_dir, img_id + ".jpg")
        xml_src = os.path.join(ann_dir, img_id + ".xml")

        jpg_dst = os.path.join(out_img_dir, img_id + ".jpg")
        txt_dst = os.path.join(out_lbl_dir, img_id + ".txt")

        if not os.path.exists(jpg_dst):
            shutil.copy(jpg_src, jpg_dst)

        convert_annotation(xml_src, txt_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Pascal VOC dataset to YOLO format."
    )

    parser.add_argument(
        "--voc-train",
        type=str,
        default="VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007",
        help="Path to the VOC Train/Val root directory",
    )
    parser.add_argument(
        "--voc-test",
        type=str,
        default="VOCtest_06-Nov-2007/VOCdevkit/VOC2007",
        help="Path to the VOC Test root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="VOC_YOLO",
        help="Directory to save the converted YOLO dataset",
    )

    args = parser.parse_args()

    out_dir = args.output_dir
    voc_train = args.voc_train
    voc_test = args.voc_test

    # Create output directories
    os.makedirs(f"{out_dir}/images/train", exist_ok=True)
    os.makedirs(f"{out_dir}/images/val", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{out_dir}/labels/val", exist_ok=True)

    print(f"Converting trainval split from: {voc_train}")
    process_split(
        voc_train, "trainval", f"{out_dir}/images/train", f"{out_dir}/labels/train"
    )

    print(f"Converting test split from: {voc_test}")
    process_split(voc_test, "test", f"{out_dir}/images/val", f"{out_dir}/labels/val")

    print(f"Writing dataset.yaml to {out_dir}...")
    yaml_content = f"""
path: {os.path.abspath(out_dir)}
train: images/train
val: images/val
nc: {len(CLASSES)}

names:
"""
    for i, c in enumerate(CLASSES):
        yaml_content += f"  {i}: {c}\n"

    with open(f"{out_dir}/dataset.yaml", "w") as f:
        f.write(yaml_content)

    print("Done!")


if __name__ == "__main__":
    main()

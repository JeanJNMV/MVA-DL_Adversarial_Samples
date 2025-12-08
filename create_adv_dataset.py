import torch
import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Adversarial YOLO Dataset")

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00,
        help="Strength of the attack (default: 0.00)",
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="both",
        choices=["train", "val", "both"],
        help="Splits to process: 'train', 'val', or 'both'",
    )

    parser.add_argument(
        "--base_dataset",
        type=str,
        default="VOC_YOLO",
        help="Path to the source dataset (default: VOC_YOLO)",
    )

    return parser.parse_args()


# Constants
MODEL_PATH = "models/yolo11n.pt"
INPUT_SIZE = 640
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


# FGSM Attack
def load_image_torch(path, input_size=640):
    """Loads image, converts to RGB, and makes it a Tensor."""
    img_raw_bgr = cv2.imread(str(path))
    if img_raw_bgr is None:
        return None, None

    img_resized_bgr = cv2.resize(img_raw_bgr, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

    original_shape = img_raw_bgr.shape[:2]
    return original_shape, img_tensor


def fgsm_attack(model_wrapper, img_tensor, epsilon):
    """Performs the FGSM attack."""
    # If epsilon is 0, no need to compute gradients
    if epsilon == 0.0:
        return img_tensor

    pytorch_model = model_wrapper.model
    img_tensor.requires_grad = True
    pytorch_model.eval()

    preds = pytorch_model(img_tensor)

    # Handle output structure
    if isinstance(preds, (list, tuple)):
        output = preds[0]
    else:
        output = preds

    loss = -torch.mean(torch.abs(output))

    pytorch_model.zero_grad()
    loss.backward()

    data_grad = img_tensor.grad.data

    # Create adversarial image
    perturbed_img = img_tensor + epsilon * data_grad.sign()
    perturbed_img = torch.clamp(perturbed_img, 0, 1)

    return perturbed_img.detach().contiguous()


def save_adversarial_image(tensor, output_path, original_shape):
    """Converts tensor back to BGR image, resizes to original, and saves."""
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_final = cv2.resize(img_bgr, (original_shape[1], original_shape[0]))

    cv2.imwrite(str(output_path), img_final)


# main function
def main():
    args = parse_arguments()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Running on {device} with Epsilon={args.epsilon}...")

    # 2. Load Model
    # Ensure the model file exists or let Ultralytics handle download/path
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
    except Exception as e:
        print(
            f"Error loading model from {MODEL_PATH}. Ensure the path is correct.\n{e}"
        )
        return

    # Format: adv_VOC_YOLO_eps_{epsilon:.2f}
    output_root = f"adv_{args.base_dataset}_eps_{args.epsilon:.2f}"

    all_splits = ["train", "val"]
    # Define which splits we actually process
    if args.splits == "both":
        process_splits = ["train", "val"]
    else:
        process_splits = [args.splits]

    # Create directory structure for all splits
    for split in all_splits:
        os.makedirs(os.path.join(output_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_root, "labels", split), exist_ok=True)

    total_processed = 0

    for split in process_splits:
        src_img_dir = Path(args.base_dataset) / "images" / split
        src_lbl_dir = Path(args.base_dataset) / "labels" / split

        dst_img_dir = Path(output_root) / "images" / split
        dst_lbl_dir = Path(output_root) / "labels" / split

        if not src_img_dir.exists():
            print(f"Warning: Source directory {src_img_dir} does not exist. Skipping.")
            continue

        image_files = (
            list(src_img_dir.glob("*.[jJ][pP][gG]"))
            + list(src_img_dir.glob("*.[jJ][pP][eE][gG]"))
            + list(src_img_dir.glob("*.[pP][nN][gG]"))
        )

        print(f"\nProcessing split: {split.upper()} ({len(image_files)} images)")

        for img_path in tqdm(image_files, desc=f"Attacking {split}"):
            try:
                label_name = img_path.stem + ".txt"
                src_label_path = src_lbl_dir / label_name

                if not src_label_path.exists():
                    # Skip images without labels (ghost images)
                    continue

                orig_shape, img_tensor = load_image_torch(img_path, INPUT_SIZE)
                if img_tensor is None:
                    continue

                img_tensor = img_tensor.to(device)

                adv_tensor = fgsm_attack(model, img_tensor, args.epsilon)

                dst_img_path = dst_img_dir / img_path.name
                save_adversarial_image(adv_tensor, dst_img_path, orig_shape)

                dst_label_path = dst_lbl_dir / label_name
                shutil.copy(src_label_path, dst_label_path)

                total_processed += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    yaml_content = f"""path: {os.path.abspath(output_root)}
train: images/train
val: images/val
nc: {len(CLASSES)}
names:
"""
    for i, cls in enumerate(CLASSES):
        yaml_content += f"  {i}: {cls}\n"

    yaml_path = Path(output_root) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Total images processed: {total_processed}")
    print(f"Dataset saved to: {output_root}")


if __name__ == "__main__":
    main()

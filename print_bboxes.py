import os
import cv2
import argparse
import torch
from ultralytics import YOLO
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run YOLO detection on a folder of images."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo11n.pt",
        help="Path to the YOLO model (default: models/yolo11n.pt).",
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        default="adv_VOC_YOLO_eps_0.00/images/val",
        help="Path to the input folder containing images.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5).",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Path to the output folder. Default: predicted_bboxes/{dataset}/{split}/{model_name}",
    )

    return parser.parse_args()


def run_detection(input_folder, confidence, model_path, output_folder=None):
    # Build default output folder: predictions/{dataset}/{split}/{model_name}
    if output_folder is None:
        # Extract dataset name and split from input_folder (e.g., "adv_VOC_YOLO_eps_0.00/images/val")
        path_parts = input_folder.rstrip(os.sep).split(os.sep)
        split = path_parts[-1] if path_parts else "val"
        dataset = path_parts[0] if path_parts else "dataset"
        # Extract model name without extension
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_folder = os.path.join("predicted_bboxes", dataset, split, model_name)
    os.makedirs(output_folder, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model ({model_path}) on {device}...")

    try:
        model = YOLO(model_path)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    image_files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print(f"No images found in '{input_folder}'.")
        return

    print(
        f"Found {len(image_files)} images. Starting detection with conf={confidence}..."
    )

    for file_name in tqdm(image_files, desc="Detecting"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f"det_{file_name}")

        try:
            results = model.predict(input_path, conf=confidence, verbose=False)

            annotated_frame = results[0].plot()

            cv2.imwrite(output_path, annotated_frame)

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

    print(f"\nDetected images saved to: '{output_folder}/'")


if __name__ == "__main__":
    args = parse_arguments()
    run_detection(args.input_folder, args.conf, args.model, args.output_folder)

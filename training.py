import argparse
import os
from ultralytics import YOLO


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n on VOC with optional Epsilon tag. MUST BE RUN ON CUDA/GOOGLE COLAB!"
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.00,
        help="Epsilon value for naming and tracking (default: 0.00)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    os.makedirs("models", exist_ok=True)

    model = YOLO("models/yolo11n.pt")

    print(f"Starting training with Epsilon identifier: {args.epsilon:.2f}")

    model.train(
        data=f"adv_VOC_YOLO_eps_{args.epsilon:.2f}/dataset.yaml",
        epochs=30,
        imgsz=640,
        batch=32,
        workers=2,
        device="cuda",
        lr0=0.001,
        half=True,
        project="runs/train",
        name=f"yolo11n_voc_eps_{args.epsilon:.2f}",
    )

    save_path = f"models/yolo11n_{args.epsilon:.2f}.pt"
    model.save(save_path)

    print(f"Training complete. Model saved to: {save_path}")


if __name__ == "__main__":
    main()

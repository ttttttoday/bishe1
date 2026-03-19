import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train vehicle detector for campus violation system.")
    parser.add_argument("--data", default="data/visdrone_vehicle.yaml", help="Dataset yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained weight or yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="cuda device id or cpu")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--project", default="runs", help="Train output root")
    parser.add_argument("--name", default="campus_vehicle_yolov8n", help="Experiment name")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="auto",
        patience=20,
    )


if __name__ == "__main__":
    main()

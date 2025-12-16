from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (.pt file)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="Path to data.yaml file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--device", default=0,
                        help="Device to use (cpu or GPU id = 0)")
    
    args = parser.parse_args()

    print(f"\n Loading model: {args.model}")
    model = YOLO(args.model)



    print(f" Running evaluation on `{args.split}` set...\n")
    metrics = model.val(
        data=args.data,
        split=args.split,
        device=args.device,
        conf=0.001,  # low threshold to evaluate more detections
        save_json=True,  # saves COCO-style metrics
        project="runs/test",
        name="custom_yolo8",
        plots=True  # generates confusion matrix + PR curve
    )

    print("\n✅ Evaluation Complete!")
    print("------------------------------------------")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print(f"mAP50:         {metrics.box.map50:.4f}")
    print(f"mAP50-95:      {metrics.box.map:.4f}")
    print("------------------------------------------")
    print("Confusion matrix, PR curve and other plots saved in:")
    print(f"runs/test/custom_yolo8")

if __name__ == "__main__":
    main()

from ultralytics import YOLO
import os
import argparse
from multiprocessing import freeze_support


#paths
DATA_YAML = 'dataset/data.yaml'
SAVE_DIR = 'runs/train/custom_yolo8'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train YOLOv8 model (safe on Windows)')
    parser.add_argument('--data', default=DATA_YAML, help='Path to data.yaml')
    parser.add_argument('--save-dir', default=SAVE_DIR, help='Directory to save runs')
    parser.add_argument('--model', default='yolov8n.pt', help='Base model or weights file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', default=0, help='Device to use (cpu or GPU id like 0)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--augment', action='store_true', help='Enable augmentation')
    parser.add_argument('--dry-run', action='store_true', help='Print config and exit (no training)')
    parser.add_argument('--name', default='yolo8_custom_model', help='Experiment name')

    args = parser.parse_args(argv)

    # create save directory if not exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # show configuration and exit if dry-run
    if args.dry_run:
        print('Dry run — no training started. Configuration:')
        print('  data:', args.data)
        print('  save_dir:', args.save_dir)
        print('  model:', args.model)
        print('  epochs:', args.epochs)
        print('  batch:', args.batch)
        print('  imgsz:', args.imgsz)
        print('  device:', args.device)
        print('  patience:', args.patience)
        print('  augment:', args.augment)
        print('  name:', args.name)
        return

    # load model
    model = YOLO(args.model)

    # train model
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=args.save_dir,
        patience=args.patience,
        augment=args.augment,
        name=args.name
    )

    print('Training complete!')
    print(f'Trained weights are saved in {args.save_dir}/yolo8_custom_model/weights/best.pt')


if __name__ == '__main__':
    freeze_support()
    main()
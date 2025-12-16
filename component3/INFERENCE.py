import argparse
import os
import sys
import cv2
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 inference using ultralytics')
    parser.add_argument('--source', '-s', default='https://ultralytics.com/images/bus.jpg',
                        help='Path or URL to image / video')
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument('--device', default=0, help='Device (cpu or GPU id like 0)')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save annotated results to ./outputs')
    args = parser.parse_args()

    # Create output directory if saving
    out_dir = os.path.abspath('outputs')
    if args.save and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f'Loading model {args.model} (device={args.device or "auto"})')
    model = YOLO(args.model)

    print('Running prediction...')
    results = model.predict(source=args.source, device=args.device, imgsz=args.imgsz, conf=args.conf)

    if not results:
        print('No results returned.')
        return

    # List to store annotated images in memory
    annotated_images = []

    for i, r in enumerate(results):
        try:
            # Get annotated image as numpy array
            annotated = r.plot()
            annotated_images.append(annotated)
        except Exception:
            annotated = None

        source_name = os.path.basename(args.source)
        if args.source.startswith('http'):
            source_name = f'source_{i}.jpg'

        # Save annotated image if requested
        if args.save and annotated is not None:
            out_path = os.path.join(out_dir, f'result_{i}_{source_name}')
            if annotated.shape[2] == 3:
                annotated_bgr = annotated[:, :, ::-1]  # Convert RGB to BGR for OpenCV
            else:
                annotated_bgr = annotated
            if not out_path.lower().endswith(('.jpg', '.png')):
                out_path += '.jpg'
            cv2.imwrite(out_path, annotated_bgr)
            print('Saved annotated output ->', out_path)

    print('Done — prediction completed.')

    # Return or use annotated images as needed
    return annotated_images

if __name__ == '__main__':
    images = main()
    if images:
        # Example: show the first annotated image in a window
        cv2.imshow("Annotated Image", images[0][:, :, ::-1])  # Convert RGB to BGR for display
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import argparse
from PIL import Image
import os
from collections import defaultdict

# Set paths for ground truth and predictions
parser = argparse.ArgumentParser(description="Evaluate YOLO predictions against ground truth labels.")
parser.add_argument("--gt", type=str, required=True, help="Ground truth labels")
parser.add_argument("--pred", type=str, required=True, help="Prediction labels")
parser.add_argument("--images", type=str, required=True, help="Images")
args = parser.parse_args()

# Set paths for ground truth and predictions
GROUND_TRUTH_PATH = args.gt
PREDICTIONS_PATH = args.pred
IMAGES_PATH = args.images
IOU_THRESHOLD = args.iou

# Threshold to classify TP/FP
IOU_THRESHOLD = 0.3  

def load_yolo_labels(file_path, is_prediction=False):
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            cls = int(parts[0])
            if is_prediction:
                x_min, y_min, x_max, y_max, confidence = map(float, parts[1:])
                labels.append((cls, x_min, y_min, x_max, y_max, confidence))
            else:
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append((cls, x_center, y_center, width, height))
    print(f"Loaded {'predictions' if is_prediction else 'ground truth'} from {file_path}: {labels}")
    return labels

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        print(f"Image dimensions for {image_path}: {img.width}x{img.height}")
        return img.width, img.height

def yolo_to_pixel(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_min = int((x_center - width / 2) * img_width)
    x_max = int((x_center + width / 2) * img_width)
    y_min = int((y_center - height / 2) * img_height)
    y_max = int((y_center + height / 2) * img_height)
    print(f"Converted YOLO box {box} to pixels: {(x_min, y_min, x_max, y_max)}")
    return x_min, y_min, x_max, y_max

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        print(f"No overlap between {box1} and {box2}")
        return 0.0  # No overlap
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    iou = intersection / union
    print(f"IoU between {box1} and {box2}: {iou:.4f}")
    return iou

def evaluate_image(gt_boxes, pred_boxes, img_width, img_height, iou_threshold, file_name):
    """Evaluate a single image for TP, FP, FN, split by class."""
    tp_per_class = defaultdict(int)
    fp_per_class = defaultdict(int)
    fn_per_class = defaultdict(int)
    fn_files = defaultdict(list)  # To store file names for false negatives
    gt_matched = set()

    for pred in pred_boxes:
        pred_cls, *pred_pixel = pred[:5]  # Predictions are already in pixel format
        matched = False

        for i, gt in enumerate(gt_boxes):
            gt_cls, *gt_box = gt
            if gt_cls != pred_cls or i in gt_matched:
                continue

            gt_pixel = yolo_to_pixel(gt_box, img_width, img_height)  # Convert ground truth boxes only
            iou = calculate_iou(pred_pixel, gt_pixel)
            if iou >= iou_threshold:
                tp_per_class[pred_cls] += 1
                matched = True
                gt_matched.add(i)
                break

        if not matched:
            fp_per_class[pred_cls] += 1

    for i, gt in enumerate(gt_boxes):
        if i not in gt_matched:
            fn_per_class[gt[0]] += 1
            fn_files[gt[0]].append(file_name)

    return tp_per_class, fp_per_class, fn_per_class, fn_files


def evaluate_dataset(gt_path, pred_path, img_path, iou_threshold):
    files = [f for f in os.listdir(gt_path) if f.endswith(".txt")]
    print(f"Evaluating dataset with {len(files)} files")
    total_tp = defaultdict(int)
    total_fp = defaultdict(int)
    total_fn = defaultdict(int)
    false_negative_files = defaultdict(list)  # Track false negative filenames

    for file in files:
        gt_file = os.path.join(gt_path, file)
        pred_file = os.path.join(pred_path, file)
        image_file = os.path.join(img_path, file.replace(".txt", ".jpg"))

        img_width, img_height = get_image_dimensions(image_file)
        gt_boxes = load_yolo_labels(gt_file)
        pred_boxes = load_yolo_labels(pred_file, is_prediction=True)

        tp, fp, fn, fn_files = evaluate_image(gt_boxes, pred_boxes, img_width, img_height, iou_threshold, file)
        for cls in tp:
            total_tp[cls] += tp[cls]
        for cls in fp:
            total_fp[cls] += fp[cls]
        for cls in fn:
            total_fn[cls] += fn[cls]
            false_negative_files[cls].extend(fn_files[cls])

    results = {}
    for cls in set(total_tp.keys()).union(total_fp.keys()).union(total_fn.keys()):
        precision = total_tp[cls] / (total_tp[cls] + total_fp[cls]) if (total_tp[cls] + total_fp[cls]) > 0 else 0
        recall = total_tp[cls] / (total_tp[cls] + total_fn[cls]) if (total_tp[cls] + total_fn[cls]) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results[cls] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "TP": total_tp[cls],
            "FP": total_fp[cls],
            "FN": total_fn[cls],
            "False Negative Files": false_negative_files[cls],  # Include file names
        }
    print(f"Total evaluation results: {results}")
    return results


# Run evaluation
results = evaluate_dataset(GROUND_TRUTH_PATH, PREDICTIONS_PATH, IMAGES_PATH, IOU_THRESHOLD)
print("Evaluation Results Per Class:")
for cls, metrics in results.items():
    print(f"Class {cls}:")
    for key, value in metrics.items():
        if key == "False Negative Files":
            print(f"  {key}: {', '.join(value)}")
        else:
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


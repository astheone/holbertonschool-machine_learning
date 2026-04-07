#!/usr/bin/env python3
"""YOLO v3 object detection - non-max suppression."""
import numpy as np
from tensorflow import keras as K


class Yolo:
    """Uses YOLO v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize Yolo object detection.

        Args:
            model_path: path to Darknet Keras model
            classes_path: path to list of class names
            class_t: box score threshold for initial filtering
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [
                line.strip() for line in f.readlines()
            ]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Darknet model outputs for a single image.

        Args:
            outputs: list of numpy.ndarrays with model predictions
            image_size: numpy.ndarray [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape
            anchors = self.anchors[i]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cx = np.tile(cx, (grid_h, 1, anchor_boxes))
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)
            cy = np.tile(cy, (1, grid_w, anchor_boxes))

            bx = (1 / (1 + np.exp(-t_x)) + cx) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + cy) / grid_h

            pw = anchors[:, 0].reshape(1, 1, anchor_boxes)
            ph = anchors[:, 1].reshape(1, 1, anchor_boxes)

            bw = (pw * np.exp(t_w)) / input_w
            bh = (ph * np.exp(t_h)) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            conf = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(conf)

            probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes using score threshold.

        Args:
            boxes: list of numpy.ndarrays (grid_h, grid_w, anchors, 4)
            box_confidences: list of numpy.ndarrays (grid_h, grid_w,
                anchors, 1)
            box_class_probs: list of numpy.ndarrays (grid_h, grid_w,
                anchors, classes)

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            class_idx = np.argmax(scores, axis=-1)
            class_score = np.max(scores, axis=-1)

            mask = class_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(class_idx[mask])
            box_scores.append(class_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression to filtered boxes.

        Args:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?)

        Returns:
            tuple of (box_predictions, predicted_box_classes,
                predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            mask = box_classes == cls
            cls_boxes = filtered_boxes[mask]
            cls_scores = box_scores[mask]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            keep = []
            while len(cls_boxes) > 0:
                keep.append(0)
                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter = inter_w * inter_h

                area0 = (
                    (cls_boxes[0, 2] - cls_boxes[0, 0]) *
                    (cls_boxes[0, 3] - cls_boxes[0, 1])
                )
                areas = (
                    (cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                    (cls_boxes[1:, 3] - cls_boxes[1:, 1])
                )
                union = area0 + areas - inter
                iou = inter / union

                cls_boxes = cls_boxes[1:][iou < self.nms_t]
                cls_scores = cls_scores[1:][iou < self.nms_t]

            box_predictions.append(cls_boxes[:len(keep)] if len(
                keep) == 1 else cls_boxes)
            predicted_box_classes.append(
                np.full(len(keep), cls)
            )
            predicted_box_scores.append(cls_scores[:len(keep)])

        box_predictions = np.concatenate(box_predictions, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes, axis=0
        )
        predicted_box_scores = np.concatenate(
            predicted_box_scores, axis=0
        )

        return box_predictions, predicted_box_classes, predicted_box_scores

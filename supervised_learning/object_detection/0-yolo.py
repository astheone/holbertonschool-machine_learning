#!/usr/bin/env python3
"""YOLO v3 object detection - initialization."""
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

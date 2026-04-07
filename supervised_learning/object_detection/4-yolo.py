#!/usr/bin/env python3
"""Yolo class with image loading method"""

import os
import cv2


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t,
                 nms_t, anchors):
        """Initialize Yolo"""
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [c.strip() for c in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def load_images(folder_path):
        """Load images from a folder"""
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
                    image_paths.append(path)

        return (images, image_paths)

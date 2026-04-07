#!/usr/bin/env python3
"""Yolo class with image preprocessing"""

import os
import cv2
import numpy as np


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

    def preprocess_images(self, images):
        """Preprocess images for YOLO"""
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for img in images:
            h, w = img.shape[:2]
            image_shapes.append([h, w])

            resized = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            normalized = resized / 255.0
            pimages.append(normalized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

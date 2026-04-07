#!/usr/bin/env python3
"""Yolo class with show_boxes method"""

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

    def show_boxes(self, image, boxes, box_classes,
                   box_scores, file_name):
        """Display image with bounding boxes"""
        img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            class_name = self.class_names[box_classes[i]]
            score = round(box_scores[i], 2)
            text = f"{class_name} {score:.2f}"

            cv2.putText(
                img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, img)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, img)

        cv2.destroyAllWindows()

import cv2
import numpy as np


class ReIDExtractor:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        self.roi_width = 64
        self.roi_height = 128
        
        self.roi_means = np.array([0.485, 0.456, 0.406]) * 255.0
        self.roi_stds = np.array([0.229, 0.224, 0.225]) * 255.0

    def preprocess_patch(self, im_crop):
        roi_input = cv2.resize(im_crop, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = (np.asarray(roi_input).astype(np.float32) - self.roi_means) / self.roi_stds
        roi_input = np.moveaxis(roi_input, -1, 0)
        object_patch = roi_input.astype('float32')
        return object_patch

    def extract_features(self, frame, bboxes):
        features = []
        
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                features.append(np.zeros(512, dtype=np.float32))
                continue
            
            im_crop = frame[y:y2, x:x2]
            
            if im_crop.size == 0:
                features.append(np.zeros(512, dtype=np.float32))
                continue
            
            patch = self.preprocess_patch(im_crop)
            blob = np.expand_dims(patch, axis=0)
            
            self.net.setInput(blob)
            feature = self.net.forward()
            
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            
            features.append(feature)
        
        return features

    def compute_cosine_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2)

    def compute_euclidean_distance(self, feat1, feat2):
        return np.linalg.norm(feat1 - feat2)

    def compute_normalized_similarity(self, feat1, feat2):
        distance = self.compute_euclidean_distance(feat1, feat2)
        return 1.0 / (1.0 + distance)


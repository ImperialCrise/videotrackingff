import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman_filter import KalmanFilter
from reid_extractor import ReIDExtractor


class Track:
    _id_counter = 0

    def __init__(self, bbox, frame_id, feature=None, dt=1/30):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox
        self.width = bbox[2]
        self.height = bbox[3]
        self.frames_since_update = 0
        self.hit_streak = 1
        self.start_frame = frame_id
        self.history = [(frame_id, bbox)]
        
        self.feature = feature
        self.feature_history = []
        if feature is not None:
            self.feature_history.append(feature)
        
        self.kf = KalmanFilter(dt=dt, u_x=0, u_y=0, std_acc=5, x_std_meas=1, y_std_meas=1)
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self.kf.init_state(cx, cy)

    def predict(self):
        pred_cx, pred_cy = self.kf.predict()
        pred_bbox = [
            pred_cx - self.width / 2,
            pred_cy - self.height / 2,
            self.width,
            self.height
        ]
        self.predicted_bbox = pred_bbox
        return pred_bbox

    def update(self, bbox, frame_id, feature=None):
        self.bbox = bbox
        self.width = bbox[2]
        self.height = bbox[3]
        
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        self.kf.update(cx, cy)
        
        if feature is not None:
            self.feature = feature
            self.feature_history.append(feature)
            if len(self.feature_history) > 30:
                self.feature_history.pop(0)
        
        self.frames_since_update = 0
        self.hit_streak += 1
        self.history.append((frame_id, bbox))

    def mark_missed(self):
        self.frames_since_update += 1
        self.hit_streak = 0
        self.bbox = self.predicted_bbox

    def get_state_bbox(self):
        cx, cy, _, _ = self.kf.get_state()
        return [
            cx - self.width / 2,
            cy - self.height / 2,
            self.width,
            self.height
        ]

    def get_smooth_feature(self):
        if len(self.feature_history) == 0:
            return self.feature
        return np.mean(self.feature_history, axis=0)

    @staticmethod
    def reset_id_counter():
        Track._id_counter = 0


class KalmanIOUReIDTracker:
    def __init__(self, model_path, iou_threshold=0.3, reid_threshold=0.5,
                 max_frames_missed=30, min_hits=1, dt=1/30, alpha=0.5, beta=0.5):
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.max_frames_missed = max_frames_missed
        self.min_hits = min_hits
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.tracks = []
        self.frame_count = 0
        
        self.reid = ReIDExtractor(model_path)
        
        Track.reset_id_counter()

    def compute_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def predict_tracks(self):
        predicted_bboxes = []
        for track in self.tracks:
            pred_bbox = track.predict()
            predicted_bboxes.append(pred_bbox)
        return predicted_bboxes

    def create_combined_similarity_matrix(self, predicted_bboxes, detections, det_features):
        n_tracks = len(self.tracks)
        n_detections = len(detections)

        if n_tracks == 0 or n_detections == 0:
            return np.empty((n_tracks, n_detections))

        iou_matrix = np.zeros((n_tracks, n_detections))
        reid_matrix = np.zeros((n_tracks, n_detections))

        for i, (track, pred_bbox) in enumerate(zip(self.tracks, predicted_bboxes)):
            for j, (det, det_feat) in enumerate(zip(detections, det_features)):
                iou_matrix[i, j] = self.compute_iou(pred_bbox, det)
                
                if track.feature is not None and det_feat is not None:
                    track_feat = track.get_smooth_feature()
                    reid_matrix[i, j] = self.reid.compute_normalized_similarity(track_feat, det_feat)
                else:
                    reid_matrix[i, j] = 0.0

        combined_matrix = self.alpha * iou_matrix + self.beta * reid_matrix

        return combined_matrix, iou_matrix, reid_matrix

    def associate_detections(self, predicted_bboxes, detections, det_features):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        combined_matrix, iou_matrix, reid_matrix = self.create_combined_similarity_matrix(
            predicted_bboxes, detections, det_features
        )
        cost_matrix = 1 - combined_matrix

        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        min_threshold = min(self.iou_threshold, self.reid_threshold)

        for t_idx, d_idx in zip(track_indices, det_indices):
            iou_score = iou_matrix[t_idx, d_idx]
            reid_score = reid_matrix[t_idx, d_idx]
            combined_score = combined_matrix[t_idx, d_idx]
            
            if iou_score >= self.iou_threshold or reid_score >= self.reid_threshold or combined_score >= min_threshold:
                matched_indices.append((t_idx, d_idx))
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)

        return matched_indices, unmatched_detections, unmatched_tracks

    def update(self, frame, detections, frame_id):
        self.frame_count = frame_id

        det_features = self.reid.extract_features(frame, detections) if len(detections) > 0 else []

        predicted_bboxes = self.predict_tracks()

        matched, unmatched_dets, unmatched_trks = self.associate_detections(
            predicted_bboxes, detections, det_features
        )

        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx], frame_id, det_features[d_idx])

        for t_idx in unmatched_trks:
            self.tracks[t_idx].mark_missed()

        for d_idx in unmatched_dets:
            feat = det_features[d_idx] if d_idx < len(det_features) else None
            new_track = Track(detections[d_idx], frame_id, feature=feat, dt=self.dt)
            self.tracks.append(new_track)

        self.tracks = [
            t for t in self.tracks
            if t.frames_since_update <= self.max_frames_missed
        ]

        active_tracks = []
        for track in self.tracks:
            if track.hit_streak >= self.min_hits or track.frames_since_update == 0:
                active_tracks.append({
                    'id': track.id,
                    'bbox': track.get_state_bbox(),
                    'predicted_bbox': track.predicted_bbox if hasattr(track, 'predicted_bbox') else track.bbox,
                    'frames_since_update': track.frames_since_update
                })

        return active_tracks

    def get_all_tracks(self):
        return self.tracks


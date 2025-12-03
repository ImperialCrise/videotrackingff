
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    _id_counter = 0

    def __init__(self, bbox, frame_id):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.bbox = bbox
        self.frames_since_update = 0
        self.hit_streak = 1
        self.start_frame = frame_id
        self.history = [(frame_id, bbox)]

    def update(self, bbox, frame_id):
        self.bbox = bbox
        self.frames_since_update = 0
        self.hit_streak += 1
        self.history.append((frame_id, bbox))

    def mark_missed(self):
        self.frames_since_update += 1
        self.hit_streak = 0

    @staticmethod
    def reset_id_counter():
        Track._id_counter = 0


class IOUTracker:
    def __init__(self, iou_threshold=0.3, max_frames_missed=30, min_hits=1):
        """
        Args:
            iou_threshold: Minimum IoU to consider a valid association
            max_frames_missed: Maximum frames a track can go without detection before removal
            min_hits: Minimum hits before a track is confirmed
        """
        self.iou_threshold = iou_threshold
        self.max_frames_missed = max_frames_missed
        self.min_hits = min_hits
        self.tracks = []
        self.frame_count = 0
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

    def create_similarity_matrix(self, detections):
        n_tracks = len(self.tracks)
        n_detections = len(detections)

        if n_tracks == 0 or n_detections == 0:
            return np.empty((n_tracks, n_detections))

        similarity_matrix = np.zeros((n_tracks, n_detections))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                similarity_matrix[i, j] = self.compute_iou(track.bbox, det)

        return similarity_matrix

    def associate_detections(self, detections):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        similarity_matrix = self.create_similarity_matrix(detections)
        cost_matrix = 1 - similarity_matrix

        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            if similarity_matrix[t_idx, d_idx] >= self.iou_threshold:
                matched_indices.append((t_idx, d_idx))
                if d_idx in unmatched_detections:
                    unmatched_detections.remove(d_idx)
                if t_idx in unmatched_tracks:
                    unmatched_tracks.remove(t_idx)

        return matched_indices, unmatched_detections, unmatched_tracks

    def update(self, detections, frame_id):
        self.frame_count = frame_id

        matched, unmatched_dets, unmatched_trks = self.associate_detections(detections)

        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx], frame_id)

        for t_idx in unmatched_trks:
            self.tracks[t_idx].mark_missed()

        for d_idx in unmatched_dets:
            new_track = Track(detections[d_idx], frame_id)
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
                    'bbox': track.bbox,
                    'frames_since_update': track.frames_since_update
                })

        return active_tracks

    def get_all_tracks(self):
        return self.tracks


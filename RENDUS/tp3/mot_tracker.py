import os
import cv2
import numpy as np
from collections import defaultdict
from kalman_iou_tracker import KalmanIOUTracker


def load_detections(det_file):
    detections = defaultdict(list)
    
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                parts = line.strip().split(',')
            
            if len(parts) >= 7:
                frame_id = int(float(parts[0]))
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                conf = float(parts[6])
                
                detections[frame_id].append([bb_left, bb_top, bb_width, bb_height, conf])
    
    return detections


def generate_colors(n_colors):
    np.random.seed(42)
    colors = []
    for _ in range(n_colors):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        colors.append(color)
    return colors


def main():
    sequence_path = '../../ADL-Rundle-6'
    det_file = os.path.join(sequence_path, 'det', 'Yolov5l', 'det.txt')
    img_dir = os.path.join(sequence_path, 'img1')
    output_dir = './output'
    
    os.makedirs(output_dir, exist_ok=True)
    
    detections = load_detections(det_file)
    print(f"Loaded detections for {len(detections)} frames")
    
    tracker = KalmanIOUTracker(
        iou_threshold=0.3,
        max_frames_missed=30,
        min_hits=1,
        dt=1/30
    )
    
    colors = generate_colors(100)
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    if len(img_files) == 0:
        print("No images found!")
        return
    
    first_img = cv2.imread(os.path.join(img_dir, img_files[0]))
    height, width = first_img.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter(
        os.path.join(output_dir, 'ADL-Rundle-6.mp4'),
        fourcc,
        30.0,
        (width, height)
    )
    
    all_results = []
    
    for img_file in img_files:
        frame_id = int(img_file.split('.')[0])
        
        frame = cv2.imread(os.path.join(img_dir, img_file))
        if frame is None:
            continue
        
        frame_dets = detections.get(frame_id, [])
        det_bboxes = [d[:4] for d in frame_dets]
        
        active_tracks = tracker.update(det_bboxes, frame_id)
        
        for track_info in active_tracks:
            track_id = track_info['id']
            bbox = track_info['bbox']
            pred_bbox = track_info['predicted_bbox']
            x, y, w, h = [int(v) for v in bbox]
            px, py, pw, ph = [int(v) for v in pred_bbox]
            
            color = colors[track_id % len(colors)]
            
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 0, 0), 1)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"ID: {track_id}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                color,
                -1
            )
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            all_results.append([
                frame_id,
                track_id,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                1,
                -1,
                -1,
                -1
            ])
        
        info_text = f"Frame: {frame_id} | Tracks: {len(active_tracks)} | Kalman+IoU"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, "Blue: Kalman Prediction | Color: Estimated", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        video_out.write(frame)
        
        cv2.imshow('Kalman+IoU MOT Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    video_out.release()
    cv2.destroyAllWindows()
    
    results_file = os.path.join(output_dir, 'ADL-Rundle-6.txt')
    with open(results_file, 'w') as f:
        for result in all_results:
            line = ','.join([str(v) for v in result])
            f.write(line + '\n')
    
    print(f"\nTracking complete!")
    print(f"Results saved to: {results_file}")
    print(f"Video saved to: {os.path.join(output_dir, 'ADL-Rundle-6.mp4')}")
    print(f"Total unique tracks: {len(set([r[1] for r in all_results]))}")


if __name__ == "__main__":
    main()


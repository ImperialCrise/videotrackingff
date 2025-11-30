"""
Main object tracking script using Kalman Filter.
Tracks a single object (SOT) through video frames.
"""

import cv2
import numpy as np
from Detector import detect
from KalmanFilter import KalmanFilter


def main():
    kf = KalmanFilter(
        dt=0.1,
        u_x=1,
        u_y=1,
        std_acc=1,
        x_std_meas=0.1,
        y_std_meas=0.1
    )
    
    cap = cv2.VideoCapture('video/randomball.avi')
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    trajectory = []
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        centers = detect(frame)
        
        predicted = kf.predict()
        
        if len(centers) > 0:
            centroid = centers[0]
            
            estimated = kf.update(centroid)
            
            trajectory.append((int(estimated[0][0]), int(estimated[1][0])))
            
            cv2.circle(
                frame,
                (int(centroid[0][0]), int(centroid[1][0])),
                10,
                (0, 255, 0),
                2
            )
            
            cv2.rectangle(
                frame,
                (int(predicted[0][0]) - 15, int(predicted[1][0]) - 15),
                (int(predicted[0][0]) + 15, int(predicted[1][0]) + 15),
                (255, 0, 0),
                2
            )
            
            cv2.rectangle(
                frame,
                (int(estimated[0][0]) - 15, int(estimated[1][0]) - 15),
                (int(estimated[0][0]) + 15, int(estimated[1][0]) + 15),
                (0, 0, 255),
                2
            )
        else:
            trajectory.append((int(predicted[0][0]), int(predicted[1][0])))
            
            cv2.rectangle(
                frame,
                (int(predicted[0][0]) - 15, int(predicted[1][0]) - 15),
                (int(predicted[0][0]) + 15, int(predicted[1][0]) + 15),
                (255, 0, 0),
                2
            )
        
        if len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                cv2.line(
                    frame,
                    trajectory[i - 1],
                    trajectory[i],
                    (255, 255, 0),
                    2
                )
        
        cv2.imshow('Object Tracking with Kalman Filter', frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


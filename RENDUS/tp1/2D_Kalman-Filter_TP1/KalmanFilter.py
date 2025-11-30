import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        Initialize Kalman Filter parameters.
        
        Args:
            dt: Sampling time (time for one cycle)
            u_x: Acceleration in x-direction
            u_y: Acceleration in y-direction
            std_acc: Process noise magnitude (standard deviation of acceleration)
            x_std_meas: Standard deviation of measurement in x-direction
            y_std_meas: Standard deviation of measurement in y-direction
        """
        self.dt = dt
        
        self.u = np.array([[u_x], [u_y]])
        
        self.x = np.array([[0], [0], [0], [0]])
        
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.B = np.array([
            [(dt**2) / 2, 0],
            [0, (dt**2) / 2],
            [dt, 0],
            [0, dt]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.Q = np.array([
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2]
        ]) * (std_acc**2)
        
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ])
        
        self.P = np.eye(self.A.shape[0])

    def predict(self):
        """
        Predict the next state estimate and error covariance.
        Time update (prediction) step of the Kalman filter.
        
        Returns:
            Predicted state estimate (x, y position)
        """
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x[:2]

    def update(self, z):
        """
        Update the state estimate with measurement.
        Measurement update (correction) step of the Kalman filter.
        
        Args:
            z: Measurement vector [x, y] (centroid coordinates)
            
        Returns:
            Updated state estimate (x, y position)
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.x[:2]


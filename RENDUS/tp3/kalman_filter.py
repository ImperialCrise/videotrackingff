import numpy as np


class KalmanFilter:
    def __init__(self, dt=0.1, u_x=0, u_y=0, std_acc=1, x_std_meas=0.1, y_std_meas=0.1):
        self.dt = dt
        
        self.u = np.array([[u_x], [u_y]])
        
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]])
        
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        
        self.B = np.array([
            [(dt**2) / 2, 0],
            [0, (dt**2) / 2],
            [dt, 0],
            [0, dt]
        ], dtype=np.float64)
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        self.Q = np.array([
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2]
        ], dtype=np.float64) * (std_acc**2)
        
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ], dtype=np.float64)
        
        self.P = np.eye(4, dtype=np.float64) * 1000

    def init_state(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, z_x, z_y):
        z = np.array([[z_x], [z_y]], dtype=np.float64)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        I = np.eye(self.H.shape[1])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x[0, 0], self.x[1, 0]

    def get_state(self):
        return self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]


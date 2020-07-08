# MIT License
#
# Copyright (c) 2020 Matthew Hampsey

import numpy as np
from pyquaternion import Quaternion
from util import skewSymmetric, quatToMatrix

#state vector:
# [0:3] orientation error
# [3:6] velocity error
# [6:9] position error
# [9:12] gyro bias
# [12:15] accelerometer bias
class Kalman:

    def __init__(self, initial_est, estimate_covariance, process_covariance, observation_covariance):
        self.estimate = initial_est
        self.estimate_covariance = estimate_covariance*np.identity(15, dtype=float)
        self.process_covariance = process_covariance*np.identity(15, dtype=float)
        self.observation_covariance = observation_covariance*np.identity(3, dtype=float)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])

        self.G = np.zeros(shape=(15, 15), dtype=float)
        self.G[0:3, 9:12] = -np.identity(3)
        self.G[6:9, 3:6] =  np.identity(3)
        
    def update(self, gyro_meas, acc_meas, time_delta):
        
        gyro_meas = gyro_meas - self.gyro_bias
        acc_meas = acc_meas - self.accelerometer_bias

        #Integrate angular velocity through forming quaternion derivative 
        self.estimate = self.estimate + time_delta*0.5*self.estimate*Quaternion(scalar = 0, vector=gyro_meas)
        self.estimate = self.estimate.normalised

        #Form process model
        self.G[0:3, 0:3] = -skewSymmetric(gyro_meas)
        self.G[3:6, 0:3] = -quatToMatrix(self.estimate).dot(skewSymmetric(acc_meas))
        self.G[3:6, 12:15] = -quatToMatrix(self.estimate)
        F = np.identity(15, dtype=float) + self.G*time_delta

        #Update with a priori covariance
        self.estimate_covariance = np.dot(np.dot(F, self.estimate_covariance), F.transpose()) + self.process_covariance

        #Form Kalman gain
        H = np.zeros(shape=(3,15), dtype=float)
        H[0:3, 0:3] = skewSymmetric(self.estimate.inverse.rotate(np.array([0.0, 0.0, -1.0])))
        H[0:3, 12:15] = np.identity(3, dtype=float)
        PH_T = np.dot(self.estimate_covariance, H.transpose())
        inn_cov = H.dot(PH_T) + self.observation_covariance
        K = np.dot(PH_T, np.linalg.inv(inn_cov))

        #Update with a posteriori covariance
        self.estimate_covariance = (np.identity(15) - np.dot(K, H)).dot(self.estimate_covariance)
        
        aposteriori_state = np.dot(K, (acc_meas - self.estimate.inverse.rotate(np.array([0.0, 0.0, -1.0]))))

        #Fold filtered error state back into full state estimates
        self.estimate = self.estimate * Quaternion(scalar = 1, vector = 0.5*aposteriori_state[0:3])
        self.estimate = self.estimate.normalised
        self.gyro_bias += aposteriori_state[9:12]
        self.accelerometer_bias += aposteriori_state[12:15]

# MIT License

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
# [15:18] magnetometer bias
class Kalman:

    def __init__(self, initial_est, estimate_covariance, 
                       gyro_cov, gyro_bias_cov, accel_proc_cov, 
                       accel_bias_cov, mag_proc_cov, mag_bias_cov, 
                       accel_obs_cov, mag_obs_cov):
        self.estimate = initial_est
        self.estimate_covariance = estimate_covariance*np.identity(18, dtype=float)
        self.observation_covariance = np.identity(6, dtype=float)
        self.observation_covariance[0:3, 0:3] = accel_obs_cov * np.identity(3, dtype=float)
        self.observation_covariance[3:6, 3:6] = mag_obs_cov * np.identity(3, dtype=float)
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.magnetometer_bias = np.array([0.0, 0.0, 0.0])

        self.G = np.zeros(shape=(18, 18), dtype=float)
        self.G[0:3, 9:12] = -np.identity(3)
        self.G[6:9, 3:6] =  np.identity(3)

        self.gyro_cov_mat = gyro_cov*np.identity(3, dtype=float)
        self.gyro_bias_cov_mat = gyro_bias_cov*np.identity(3, dtype=float)
        self.accel_cov_mat = accel_proc_cov*np.identity(3, dtype=float)
        self.accel_bias_cov_mat = accel_bias_cov*np.identity(3, dtype=float)
        self.mag_cov_mat = mag_proc_cov*np.identity(3, dtype=float)
        self.mag_bias_cov_mat = mag_bias_cov*np.identity(3, dtype=float)

    def process_covariance(self, time_delta):
        Q = np.zeros(shape=(18, 18), dtype=float)
        Q[0:3, 0:3] = self.gyro_cov_mat*time_delta + self.gyro_bias_cov_mat*(time_delta**3)/3.0
        Q[0:3, 9:12] = -self.gyro_bias_cov_mat*(time_delta**2)/2.0
        Q[3:6, 3:6] = self.accel_cov_mat*time_delta + self.accel_bias_cov_mat*(time_delta**3)/3.0
        Q[3:6, 6:9] = self.accel_bias_cov_mat*(time_delta**4)/8.0 + self.accel_cov_mat*(time_delta**2)/2.0
        Q[3:6, 12:15] = -self.accel_bias_cov_mat*(time_delta**2)/2.0
        Q[6:9, 3:6] = self.accel_cov_mat*(time_delta**2)/2.0 + self.accel_bias_cov_mat*(time_delta**4)/8.0
        Q[6:9, 6:9] = self.accel_cov_mat*(time_delta**3)/3.0 + self.accel_bias_cov_mat*(time_delta**5)/20.0
        Q[6:9, 12:15] = -self.accel_bias_cov_mat*(time_delta**3)/6.0
        Q[9:12, 0:3] = -self.gyro_bias_cov_mat*(time_delta**2)/2.0
        Q[9:12, 9:12] = self.gyro_bias_cov_mat*time_delta
        Q[12:15, 3:6] = -self.accel_bias_cov_mat*(time_delta**2)/2.0
        Q[12:15, 6:9] = -self.accel_bias_cov_mat*(time_delta**3)/6.0
        Q[12:15, 12:15] = self.accel_bias_cov_mat*time_delta
        Q[15:18, 15:18] = self.mag_bias_cov_mat*time_delta

        return Q
        
    def update(self, gyro_meas, acc_meas, mag_meas, time_delta):
        
        gyro_meas = gyro_meas - self.gyro_bias
        acc_meas = acc_meas - self.accelerometer_bias
        mag_meas = mag_meas - self.magnetometer_bias

        #Integrate angular velocity through forming quaternion derivative
        self.estimate = self.estimate + time_delta*0.5*self.estimate*Quaternion(scalar = 0, vector=gyro_meas)
        self.estimate = self.estimate.normalised
        
        #Form process model
        self.G[0:3, 0:3] = -skewSymmetric(gyro_meas)
        self.G[3:6, 0:3] = -quatToMatrix(self.estimate).dot(skewSymmetric(acc_meas))
        self.G[3:6, 12:15] = -quatToMatrix(self.estimate)
        F = np.identity(18, dtype=float) + self.G*time_delta

        #Update with a priori covariance
        self.estimate_covariance = np.dot(np.dot(F, self.estimate_covariance), F.transpose()) + self.process_covariance(time_delta)

        #Form Kalman gain
        H = np.zeros(shape=(6,18), dtype=float)
        H[0:3, 0:3] = skewSymmetric(self.estimate.inverse.rotate(np.array([0.0, 0.0, -1.0])))
        H[0:3, 12:15] = np.identity(3, dtype=float)
        H[3:6, 0:3] = skewSymmetric(self.estimate.inverse.rotate(np.array([1.0, 0, 0])))
        H[3:6, 15:18] = np.identity(3, dtype=float)
        PH_T = np.dot(self.estimate_covariance, H.transpose())
        inn_cov = H.dot(PH_T) + self.observation_covariance
        K = np.dot(PH_T, np.linalg.inv(inn_cov))

        #Update with a posteriori covariance
        self.estimate_covariance = (np.identity(18) - np.dot(K, H)).dot(self.estimate_covariance)
        
        observation = np.zeros(shape=(6, ), dtype=float)
        observation[0:3] = acc_meas
        observation[3:6] = mag_meas
        predicted_observation = np.zeros(shape=(6, ), dtype=float)
        predicted_observation[0:3] = self.estimate.inverse.rotate(np.array([0.0, 0.0, -1.0]))
        predicted_observation[3:6] = self.estimate.inverse.rotate(np.array([1.0, 0.0, 0.0]))

        aposteriori_state = np.dot(K, (observation - predicted_observation).transpose())

        #Fold filtered error state back into full state estimates
        self.estimate = self.estimate * Quaternion(scalar = 1, vector = 0.5*aposteriori_state[0:3])
        self.estimate = self.estimate.normalised
        self.gyro_bias += aposteriori_state[9:12]
        self.accelerometer_bias += aposteriori_state[12:15]
        self.magnetometer_bias += aposteriori_state[15:18]

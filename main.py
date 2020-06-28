import numpy as np
import numpy.random as npr
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import math
import model
from referencevectorgauge import ReferenceVectorGauge
from noisydevice import NoisyDeviceDecorator
import gyro
import kalman3 as kalman

def quatToEuler(q):
    euler = []
    x = math.atan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]*q[1] + q[2]*q[2]))
    euler.append(x)

    x = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    euler.append(x)

    x = math.atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]*q[2] + q[3]*q[3]))
    euler.append(x)

    return euler

def quatListToEuler(qs):
    euler = np.ndarray(shape=(3, len(qs)), dtype=float)

    for (i, q) in enumerate(qs):
        e = quatToEuler(q)
        euler[0, i] = e[0]
        euler[1, i] = e[1]
        euler[2, i] = e[2]

    return euler

def rmse_euler(estimate, truth):
    def rmse(vec1, vec2):
        return np.sqrt(np.mean((vec1 - vec2)**2))

    return[rmse(estimate[0], truth[0]), 
           rmse(estimate[1], truth[1]),
           rmse(estimate[2], truth[2])]

if __name__ == '__main__':
    accel_cov = 0.001
    accel_bias = np.array([0.6, 0.02, 0.05])
    mag_cov = 0.001
    mag_bias = np.array([0.03, 0.08, 0.04])
    gyro_cov = 0.001
    gyro_bias = np.array([0.25, 0.01, 0.05])
    gyro_bias_drift = 0.0001
    gyro = NoisyDeviceDecorator(gyro.Gyro(), gyro_bias, gyro_cov, gyro_bias_drift)
    accelerometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([0, 0, -1])), 
            accel_bias, accel_cov, bias_drift_covariance = 0.0) 
    magnetometer = NoisyDeviceDecorator(ReferenceVectorGauge(np.array([1, 0, 0])), 
            mag_bias, mag_cov, bias_drift_covariance = 0.0) 
    real_measurement = np.array([0.0, 0.0, 0.0])
    time_delta = 0.005
    true_orientation = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    estimate = model.Model(Quaternion(axis = [1, 0, 0], angle=0))
    true_rots = []
    est_rots = []
    filtered_rots = []

    kalman = kalman.Kalman(true_orientation.orientation, 1.0, 0.1, 0.1)
    for i in range(1000):

        if (i % 10 == 0):
          real_measurement = npr.normal(0.0, 1.0, 3)

        gyro_measurement = gyro.measure(time_delta, real_measurement)

        estimate.update(time_delta, gyro_measurement)
        est_rots.append(estimate.orientation)

        true_orientation.update(time_delta, real_measurement)
        true_rots.append(true_orientation.orientation)
        
        measured_acc = accelerometer.measure(time_delta, true_orientation.orientation)
        measured_mag = magnetometer.measure(time_delta, true_orientation.orientation)

        #kalman.update(gyro_measurement, measured_acc, time_delta)
        kalman.update(gyro_measurement, measured_acc, measured_mag, time_delta)
        filtered_rots.append(kalman.estimate)

    print "gyro bias: ", kalman.gyro_bias
    print "accel bias: ", kalman.accelerometer_bias
    #print "mag bias: ", kalman.magnetometer_bias
    est_euler = quatListToEuler(est_rots)
    true_euler = quatListToEuler(true_rots)

    filtered_euler = quatListToEuler(filtered_rots)

    errors = []
    errors.append(np.minimum(np.minimum(np.abs(est_euler[0] - true_euler[0]), np.abs(2*math.pi + est_euler[0] - true_euler[0])),
                    np.abs(-2*math.pi + est_euler[0] - true_euler[0])))
    errors.append(np.minimum(np.minimum(np.abs(est_euler[1] - true_euler[1]), np.abs(2*math.pi + est_euler[1] - true_euler[1])),
                    np.abs(-2*math.pi + est_euler[1] - true_euler[1])))
    errors.append(np.minimum(np.minimum(np.abs(est_euler[2] - true_euler[2]), np.abs(2*math.pi + est_euler[2] - true_euler[2])),
                    np.abs(-2*math.pi + est_euler[2] - true_euler[2])))


    filtered_errors = []
    filtered_errors.append(np.minimum(np.minimum(np.abs(filtered_euler[0] - true_euler[0]), np.abs(2*math.pi + filtered_euler[0] - true_euler[0])),
                    np.abs(-2*math.pi + filtered_euler[0] - true_euler[0])))
    filtered_errors.append(np.minimum(np.minimum(np.abs(filtered_euler[1] - true_euler[1]), np.abs(2*math.pi + filtered_euler[1] - true_euler[1])),
                    np.abs(-2*math.pi + filtered_euler[1] - true_euler[1])))
    filtered_errors.append(np.minimum(np.minimum(np.abs(filtered_euler[2] - true_euler[2]), np.abs(2*math.pi + filtered_euler[2] - true_euler[2])),
                    np.abs(-2*math.pi + filtered_euler[2] - true_euler[2])))

    one, = plt.plot(errors[0], label='unfiltered roll')
    two, = plt.plot(errors[1], label='unfiltered pitch')
    three, = plt.plot(errors[2], label='unfiltered yaw')
    four, = plt.plot(filtered_errors[0], label='filtered roll')
    five, = plt.plot(filtered_errors[1], label='filtered pitch')
    six, = plt.plot(filtered_errors[2], label='filtered yaw')
    #plt.ylabel('some numbers')
    plt.legend(handles=[one, two, three, four, five, six])
    plt.show()

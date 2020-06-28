import numpy.random as npr


class NoisyDeviceDecorator:
    def __init__(self, measuring_device, initial_bias, noise_covariance, bias_drift_covariance):
        self.measuring_device = measuring_device
        self.bias = initial_bias
        self.noise_covariance = noise_covariance
        self.bias_drift_covariance = bias_drift_covariance

    def measure(self, time_delta, *args):
        measurement = self.measuring_device.measure(time_delta, *args)
        bias_derivative = npr.normal(0.0, self.bias_drift_covariance)
        self.bias += bias_derivative*time_delta
        noise = npr.normal(0.0, self.noise_covariance, len(measurement))

        return measurement + self.bias + noise

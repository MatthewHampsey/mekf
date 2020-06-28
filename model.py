from pyquaternion import Quaternion

class Model:
    def __init__(self, orientation = Quaternion(axis = [1, 0, 0], angle=0)):
        self.orientation = orientation

    def update(self, time_delta, gyro_measurement):
        orientation_derivative = 0.5*self.orientation*Quaternion(scalar=0, vector=gyro_measurement)
        self.orientation += time_delta*orientation_derivative
        self.orientation = self.orientation.normalised




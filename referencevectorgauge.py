import numpy as np

class ReferenceVectorGauge:
    def __init__(self, reference_vector):
        self.reference_vector = reference_vector

    def measure(self, time_delta, true_orientation):
        rotated_reference = true_orientation.inverse.rotate(self.reference_vector)
        return rotated_reference/np.sqrt(rotated_reference.dot(rotated_reference))


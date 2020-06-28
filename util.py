import numpy as np
from pyquaternion import Quaternion 

def skewSymmetric(v):
  return np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]]) 
def quatToMatrix(q):
  return 2.0*np.outer(q.vector, q.vector) \
           + np.identity(3)*(q.scalar**2 - q.vector.dot(q.vector)) \
           + 2*q.scalar*skewSymmetric(q.vector)


import numpy as np

from kinematics.dhparameters import DHParameters


class Kinematics:
    def __init__(self):
        self.joint1_dh = DHParameters(0, 0.1625, 0)
        self.joint2_dh = DHParameters(0, 0, np.pi / 2)
        self.joint3_dh = DHParameters(-0.425, 0, 0)
        self.joint4_dh = DHParameters(-0.39225, 0.1333, 0)
        self.joint5_dh = DHParameters(0, 0.0997, np.pi / 2)
        self.joint6_dh = DHParameters(0, 0.0996, -np.pi / 2)

    def compute_transformation_matrix(self, theta, dh_params):
        c = np.cos(theta)
        s = np.sin(theta)
        ca = np.cos(dh_params.alpha)
        sa = np.sin(dh_params.alpha)
        A = [[c, -s, 0, dh_params.a],
             [s*ca, c*ca, -sa, -sa*dh_params.d],
             [s*sa, c*sa, ca, ca*dh_params.d],
             [0, 0, 0, 1]]
        A = np.array(A)
        return A



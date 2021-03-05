import numpy as np

from kinematics.kinematics import Kinematics


class ForwardKinematics(Kinematics):
    def __init__(self):
        super().__init__()
        self.BRX = np.array([[1, 0, 0, 0],
                        [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2), 0],
                        [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2), 0],
                        [0, 0, 0, 1]])
        self.BRZ = np.array([[np.cos(np.pi), -np.sin(np.pi), 0, 0],
                        [np.sin(np.pi), np.cos(np.pi), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
        #self.TB0 = np.matmul(self.BRX, self.BRZ)
        self.TB0 = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        self.T6T = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        self.T0B = np.linalg.inv(self.TB0)
        self.ARUCO_TO_B = [[0.707, 0.707 * np.cos(0.5 * np.pi),  -0.707 * np.sin(0.5 * np.pi), 0],
                           [-0.707, 0.707 * np.cos(0.5 * np.pi), -0.707 * np.sin(0.5 * np.pi), 0],
                           [0, np.sin(0.5 * np.pi), np.cos(0.5 * np.pi), 0],
                           [0, 0, 0, 1]]

    def compute_0_to_6_matrix(self, thetas):
        T01 = self.compute_transformation_matrix(thetas[0], self.joint1_dh)
        T12 = self.compute_transformation_matrix(thetas[1], self.joint2_dh)
        T23 = self.compute_transformation_matrix(thetas[2], self.joint3_dh)
        T34 = self.compute_transformation_matrix(thetas[3], self.joint4_dh)
        T45 = self.compute_transformation_matrix(thetas[4], self.joint5_dh)
        T56 = self.compute_transformation_matrix(thetas[5], self.joint6_dh)
        T06 = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(T01, T12), T23), T34), T45), T56)
        return T06

    def computer_base_to_6_matrix(self, thetas):
        T06 = np.matmul(self.TB0, self.compute_0_to_6_matrix(thetas))
        return T06

    def compute_TBT(self, thetas):
        return np.matmul(self.computer_base_to_6_matrix(thetas), self.T6T)

    def convert_TBT_to_T06(self, TBT):
        TT6 = np.linalg.inv(self.T6T)
        TB6 = np.matmul(TBT, TT6)
        return self.convert_TB6_to_T06(TB6)

    def convert_T06_to_TB6(self, T06):
        return np.matmul(self.TB0, T06)

    def convert_TB6_to_T06(self, TB6):
        return np.matmul(self.T0B, TB6)

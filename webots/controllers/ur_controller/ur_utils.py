import numpy as np
from scipy.spatial.transform import Rotation

class Utils:
    @staticmethod
    def tmat_to_trans_and_rot(tmat):
        rot_mat = np.array([[tmat[0][0], tmat[0][1], tmat[0][2]],
                   [tmat[1][0], tmat[1][1], tmat[1][2]],
                   [tmat[2][0], tmat[2][1], tmat[2][2]]])
        trans = np.array([tmat[0][3], tmat[1][3], tmat[2][3]])
        return (trans, Rotation.from_matrix(rot_mat))

    @staticmethod
    def trans_and_rot_to_tmat(trans, rot: Rotation):
        rot_mat = rot.as_matrix()
        tmat = [[rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], trans[0]],
                [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], trans[1]],
                [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], trans[2]],
                [0, 0, 0, 1]]
        return np.array(tmat)

    @staticmethod
    def print_tmat(tmat, name='', format='rotvec'):
        if format == 'rotvec':
            trans, rot = Utils.tmat_to_trans_and_rot(tmat)
            if name != '':
                print(name)
            print("  translation: ", trans)
            print("  rotvec:      ", rot.as_rotvec())
        elif format == 'matrix':
            if name != '':
                print(name)
            print(np.array(tmat))
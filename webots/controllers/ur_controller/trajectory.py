from kinematics.inverse import InverseKinematics
from kinematics.forward import ForwardKinematics
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ur_utils import Utils


class Trajectory:
    def __init__(self, motor_sensors, fkin, timestep):
        self.motor_sensors = motor_sensors
        self.timestep = timestep
        self.joint_angles = [0]*6
        self.fkin = fkin
        self.ikin = InverseKinematics()
        self.rotation_matrix = [[0]*3]*3
        self.starting_rotation = None
        self.starting_cart_pos = None
        self.goal_cart_pos = None
        self.goal_rotation = None
        self.last_computed_angles = None
        self.config_id = None
        self.total_steps = 0
        self.current_step = 0
        self.slerp = None
        self.is_done = True
        import scipy
        if scipy.version.version[0] == 1 and scipy.version.version[2] < 4:
            raise Exception("Scipy >1.4.1 required")

    def _calculate_distance(self, x1, y1, z1, x2, y2, z2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def _get_joint_angles(self):
        for i in range(6):
            self.joint_angles[i] = self.motor_sensors[i].getValue()
        return self.joint_angles

    def generate_trajectory(self, goal, speed):
        self.goal_cart_pos = np.array([goal[0], goal[1], goal[2]])
        self.goal_rotation = Rotation.from_rotvec([goal[3], goal[4], goal[5]])
        transformation_matrix = self.fkin.compute_TBT(self._get_joint_angles())
        self.starting_cart_pos, self.starting_rotation = Utils.tmat_to_trans_and_rot(transformation_matrix)
        distance = self._calculate_distance(*self.starting_cart_pos, *self.goal_cart_pos)
        print("distance: " + str(distance))
        self.total_steps = int((distance/speed) / (self.timestep/1000))
        if self.total_steps == 0:
            self.total_steps = 1 # if the robot is already there execute 1 step with no move
        self.config_id = self.ikin.get_current_configuration_id(self.joint_angles)
        print(f"current config ID: {self.config_id}")
        self.last_computed_angles = self._get_joint_angles()

        Utils.print_tmat(transformation_matrix, "start")
        print("goal ", goal)

        combined_rotations = Rotation.from_quat([self.starting_rotation.as_quat(),self.goal_rotation.as_quat()])
        self.slerp = Slerp([0, 1], combined_rotations)
        self.current_step = 0
        self.is_done = False

    def calculate_step(self):
        if not self.is_done:
            self.current_step += 1
            if self.current_step % 50 == 0:
                print("step "+str(self.current_step)+"/"+str(self.total_steps))
            cart_pos = self.starting_cart_pos + (self.current_step / self.total_steps) * (self.goal_cart_pos - self.starting_cart_pos)
            r = self.slerp(self.current_step/self.total_steps)
            transformation_matrix_BT= Utils.trans_and_rot_to_tmat(cart_pos, r)
            transformation_matrix_06 = self.fkin.convert_TBT_to_T06(transformation_matrix_BT)
            if self.current_step == self.total_steps:
                self.is_done = True
            computed_angles = self.ikin.get_best_solution_for_config_id(transformation_matrix_06, self.config_id)
            computed_angles_copy = computed_angles.copy()
            for i in range(6):
                computed_angles[i] -= 6.28
                difference = abs(computed_angles[i] - self.last_computed_angles[i])
                limit = 0
                while difference > 0.5 and limit < 3:
                    computed_angles[i] += 6.28
                    difference = abs(computed_angles[i] - self.last_computed_angles[i])
                    limit += 1
                if limit >= 3:
                    print("computed, last computed")
                    print(computed_angles_copy)
                    print(self.last_computed_angles)
                    raise Exception("Something went wrong, probably singularity")
            #print(computed_angles)
            self.last_computed_angles = computed_angles
            return computed_angles
        else:
            return self.last_computed_angles
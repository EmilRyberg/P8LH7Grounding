import rospy
import actionlib
from ur_e_webots.msg import GripperAction


class GripperActionServer:
    FINGER_CLOSED_POSITION = [0.005, 0.005]  # right, left
    FINGER_OPEN_POSITION = [-0.035, 0.045]  # right, left
    MAX_FINGER_DISTANCE = 80  # mm abs open pos
    CLOSED_FINGER_DISTANCE = 10 # mm

    def __init__(self, robot, jointPrefix, nodeName):
        self.server = actionlib.ActionServer(nodeName + "gripper_server",
                                             GripperAction,
                                             self.on_goal, self.on_cancel, auto_start=False)
        self.finger_motors = [robot.getDevice("right_finger_motor"), robot.getDevice("left_finger_motor")]
        self.finger_sensors = [robot.getDevice("right_finger_sensor"), robot.getDevice("left_finger_sensor")]
        self.timestep = int(robot.getBasicTimeStep())
        self.robot = robot
        self.action = None
        self.goal_handle = None
        for sensor in self.finger_sensors:
            sensor.enable(self.timestep)

    def start(self):
        self.server.start()
        print("The gripper action server for this driver has been started")

    def on_goal(self, goal_handle):
        self.action = goal_handle.get_goal().action
        self.goal_handle = goal_handle
        self.finger_motors[1].setVelocity(1)
        self.finger_motors[1].setPosition(GripperActionServer.FINGER_CLOSED_POSITION[0])
        self.finger_motors[0].setVelocity(1)
        self.finger_motors[0].setPosition(GripperActionServer.FINGER_CLOSED_POSITION[1])
        goal_handle.set_accepted()

    def on_cancel(self, goal_handle):
        if goal_handle == self.goal_handle:
            # stop the motors
            for index, motor in enumerate(self.finger_motors):
                motor.setPosition(self.sensors[index].getValue())
            self.goal_handle.set_canceled()
            self.goal_handle = None
        else:
            goal_handle.set_canceled()

    def update(self):
        if self.robot and self.action:
            now = self.robot.getTime()
            if self.action == "close":
                if self.finger_sensors[0].getValue() - 0.0001 <= GripperActionServer.FINGER_CLOSED_POSITION[1] <= self.finger_sensors[0].getValue() + 0.0001:
                    self.action = None
                    self.goal_handle.set_succeeded()

import rospy
import actionlib
from ur_e_webots.msg import GripperAction


class GripperActionServer:
    FINGER_CLOSED_POSITION = [0.005, 0.005]  # right, left
    FINGER_OPEN_POSITION = [-0.035, 0.045]  # right, left
    MAX_FINGER_DISTANCE = 80  # mm abs open pos
    CLOSED_FINGER_DISTANCE = 10 # mm

    def __init__(self, robot, jointPrefix, nodeName, suction, gripper_connector, gripper_connector_for_box):
        self.server = actionlib.ActionServer(nodeName + "gripper_server",
                                             GripperAction,
                                             self.on_goal, self.on_cancel, auto_start=False)
        self.finger_motors = [robot.getDevice("right_finger_motor"), robot.getDevice("left_finger_motor")]
        self.finger_sensors = [robot.getDevice("right_finger_sensor"), robot.getDevice("left_finger_sensor")]
        self.timestep = int(robot.getBasicTimeStep())
        self.finger_position = (0, 0)
        self.should_lock = False
        self.should_grip_box = False
        self.goal_handle = None
        self.robot = robot
        self.suction = suction
        self.gripper_connector = gripper_connector
        self.gripper_connector_for_box = gripper_connector_for_box
        for sensor in self.finger_sensors:
            sensor.enable(self.timestep)

    def start(self):
        self.server.start()
        rospy.loginfo("The gripper action server for this driver has been started")

    def on_goal(self, goal_handle):
        self.goal_handle = goal_handle
        if self.goal_handle.get_goal().action == "close":
            width = self.goal_handle.get_goal().width
            speed = self.goal_handle.get_goal().speed
            self.should_lock = self.goal_handle.get_goal().lock
            self.should_grip_box = self.goal_handle.get_goal().grip_box
            if width > MAX_FINGER_DISTANCE:
                rospy.logwarn(f"WARNING: Width over max distance of {MAX_FINGER_DISTANCE}mm. Clamping value")
                width = MAX_FINGER_DISTANCE
            elif width < 0:
                rospy.logwarn("WARNING: Width under min distance of 0mm. Clamping value")
                width = 0
            percent_closed = (MAX_FINGER_DISTANCE - width) / MAX_FINGER_DISTANCE
            right_finger_position = GripperActionServer.FINGER_OPEN_POSITION[0] - (GripperActionServer.FINGER_OPEN_POSITION[0] - GripperActionServer.FINGER_CLOSED_POSITION[0]) * percent_closed
            left_finger_position = GripperActionServer.FINGER_OPEN_POSITION[1] - (GripperActionServer.FINGER_OPEN_POSITION[1] - GripperActionServer.FINGER_CLOSED_POSITION[1]) * percent_closed
            self.finger_position = (left_finger_position, right_finger_position)
            self.finger_motors[1].setVelocity(speed)
            self.finger_motors[1].setPosition(left_finger_position)
            self.finger_motors[0].setVelocity(speed)
            self.finger_motors[0].setPosition(right_finger_position)
        elif self.goal_handle.get_goal().action == "open":
            self.gripper_connector.unlock()
            self.gripper_connector_for_box.unlock()
            self.finger_motors[1].setVelocity(5)
            self.finger_motors[1].setPosition(GripperActionServer.FINGER_OPEN_POSITION[1])
            self.finger_motors[0].setVelocity(5)
            self.finger_motors[0].setPosition(GripperActionServer.FINGER_OPEN_POSITION[0])
        elif self.goal_handle.get_goal().action == "suction_on":
            self.suction.lock()
        elif self.goal_handle.get_goal().action == "suction_off":
            self.suction.unlock()
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
        if self.robot and self.goal_handle:
            #now = self.robot.getTime()
            action = self.goal_handle.get_goal().action
            if self.goal_handle.get_goal_status().status == actionlib_msgs.msg.GoalStatus.ACTIVE:
                if action == "close":
                    if self.finger_sensors[0].getValue() - 0.0001 <= self.finger_position[1] <= self.finger_sensors[0].getValue() + 0.0001:
                        if lock:
                            self.gripper_connector.lock()
                        if gripping_box:
                            self.gripper_connector_for_box.lock()
                        self.goal_handle.set_succeeded()
                elif action == "open":
                    if self.finger_sensors[0].getValue() - 0.0001 <= GripperActionServer.FINGER_OPEN_POSITION[1] <= self.finger_sensors[0].getValue() + 0.0001:
                        self.goal_handle.set_succeeded()
                elif action == "suction_on" or action == "suction_off":
                    self.goal_handle.set_succeeded() # could maybe just be done in the on_goal

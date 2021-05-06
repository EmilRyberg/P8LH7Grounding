from vision_lib.vision_controller import ObjectInfo
import rospy
import actionlib
from bin_picking.msg import PickObjectAction, PickObjectGoal
import cv2
from cv_bridge import CvBridge
import numpy as np 
import cv_bridge


class RobotController:
    def __init__(self):
        self.bridge = CvBridge()
        self.is_home = True
        self.client = actionlib.SimpleActionClient("pick_object", PickObjectAction)
    
    def pick_up(self, object_info: ObjectInfo, rgb, depth):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        goal.depth_img = self.bridge.cv2_to_imgmsg(depth)
        goal.command = "pick_object"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        return result.success

    def place(self, position, rgb):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.command = "place_object"
        goal.place_image_x = round(position[0])
        goal.place_image_y = round(position[1])
        goal.place_world_z = position[2]
        goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        return result.success

    def point_at(self, object_info: ObjectInfo, rgb, depth):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        goal.depth_img = self.bridge.cv2_to_imgmsg(depth)
        goal.command = "point_at"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        return result.success

    def move_out_of_view(self):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.command = "move_out_of_view"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        self.is_home = result

        return result.success

    def is_out_of_view(self):
        return not self.is_home


if __name__ == "__main__":
    rospy.init_node("controller", anonymous=True)
    controller = RobotController()
    controller.move_out_of_view()
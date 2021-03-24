import sys
import rospy
from ner.srv import NER
from little_helper_interfaces.msg import Task, ObjectEntity, OuterObjectEntity, SpatialDescription, OuterTask
from vision.vision.ros_camera_interface import ROSCamera
from speech_to_text.speechtotext import Speech_to_text
#from robot_controller.robot_controller import RobotController #TODO actually make this controller
from grounding import Grounding
"""
Man i wish we followed the uml better
"""
class DialogFlow:
    def __init__(self):
        self.stt = Speech_to_text()
        self.first_convo_flag = True
        self.grounding = Grounding()
        self.object_info = None
        self.sentence = ""
        self.robot = RobotController()
        self.camera = ROSCamera()

        #ROS Publishers
        self.tts_pub = rospy.Publisher('chatter', String, queue_size=10)
        self.find_pub = rospy.Publisher('MainFind', ObjectEntity, queue_size=10)
        self.learn_pub = rospy.Publisher('MainLearn', String, queue_size=10)

    def controller(self):
        #check to see if initialising conversation and gets sentence
        if self.first_convo_flag == True:
            self.sentence = self.first_conversation()
            self.first_convo_flag = False
        else:
            self.sentence = self.continuing_conversation()

        #subscribe to the ner service
        rospy.wait_for_service('ner')
        try:
            ner_service = rospy.ServiceProxy('ner', NER)
            task = ner_service(sentence)
        except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

        np_rgb_image = self.camera.get_image()
        np_depth_image = self.camera.get_depth()

        #TODO decide if we want the grounding node to be a service or just to use at as class normally
        try:
            grounding_service = rospy.ServiceProxy('grounding', Grounding)
            self.object_info = grounding_service(task.object1, np_rgb_image)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)

        #TODO add known flag in the return from grounding
        if object_info.known == 0:
            learn_control(task.object1.name, np_rgb_image)

        #TODO update when robot_controller is made
        else object_info.known == 1:
            if task.type == "pick":
                pick_control(object_info, np_rgb_image, np_depth_image)
            if task.type == "find":
                find_control(object_info)
            if task.type == "learn":
                learn_control(task.object1.name, np_rgb_image)

    def first_conversation(self):
        spoken_sentence = "Hello. What would you like me to do?"
        captured_audio = self.stt.Listener()
        recognized_text = self.stt.Recognizer(captured_audio)
        return recognized_text

    def continuing_conversation(self):
        spoken_sentence = "Is there anything else you want me to do?"
        captured_audio = self.stt.Listener()
        recognized_text = self.stt.Recognizer(captured_audio)
        return recognized_text        

    def learn_control(self, name, image):
        self.tts_pub.publish("I will try and learn the new object")
        self.grounding.learn_new_object(name, image)

    def pick_control(self, object_info, rgb, depth):
        self.tts_pub.publish("Okay, I will try to pick up the %s" %object_info.name) #might need to rework if we take the name out of objectinfo
        self.robot.pick(object_info, rgb, depth)

    def find_control(self, object_info, rgb):
        self.tts_pub.publish("Okay, I will try to find the %s" %object_info.name)
        self.robot.find(object_info, rgb)

if __name__ == '__main__':
    try:
        rospy.init_node('dialog_controller', anonymous=True)
        dialog = DialogFlow()
        while True:
            dialog.controller()
    except rospy.ROSInterruptException:
        pass


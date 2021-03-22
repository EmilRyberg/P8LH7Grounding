import sys
import rospy
from ner.srv import NER
from speech_to_text.speechtotext import Speech_to_text
import actionlib

class DialogFlow:
    def __init__(self):
        self.stt = Speech_to_text()
        self.first_convo_flag = True
        self.grounding = grounding()
        self.object_info = None
        self.sentence = ""
        self.tts_pub = rospy.Publisher('chatter', String, queue_size=10)
        self.find_pub = rospy.Publisher('MainFind', ObjectEntity, queue_size=10)
        self.learn_pub = rospy.Publisher('MainLearn', String, queue_size=10)
        self.robot_controller_client = actionlib.SimpleActionClient(
                                                                #'robot_controller',
                                                                #task.action
                                                                )

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

        #find object based on the ner results
        self.find_pub.publish(task.object1)  #this should be a service tbh
        rospy.Subscriber('GroundingInfo', ObjectInfo, self.grounding_cb)

        if self.object_info.known == False: #check if object is known
            self.learn_pub.publish(task.object1, task.object2)

        #logical flow for different tasks
        if task.type == "find":
            self.find_task_control(object_info)
        else if task.type == "pick_up":
            self.pick_up_task_control(object_info)

        """ possible thing to implement later
        else if task.type == "place":
            self.place_pub.publish(task.object1, location)
        """

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

    def grounding_cb(self, cb):
        self.object_info = cb

    def find_task_control(self, object_info):
        spoken_sentence = "I will try to find %s"
        #tts_pub(spoken_sentence %object_info.name)
        
    def pick_up_task_control(self, object_info):
        spoken_sentence = "I will try to pick up %s"
        #tts_pub(spoken_sentence %object_info.name)
        #robot_controller_client.send_goal(object_info)

if __name__ == '__main__':
    try:
        rospy.init_node('dialog_controller', anonymous=True)
        dialog = DialogFlow()
        while True:
            dialog.controller()
    except rospy.ROSInterruptException:
        pass


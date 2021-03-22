#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from little_helper_interfaces.msg import ObjectEntity, ObjectInfo, OuterObjectEntity
from grounding.grounding import Grounding

class GroundingNode():
    def __init__(self):
        self.infopub = rospy.Publisher('GroundingInfo', ObjectInfo, queue_size=10)
        self.grounding = Grounding()

        while not rospy.is_shutdown():
            self.listener()

    def listener(self):
        rospy.Subscriber('MainLearn', OuterObjectEntity, self.grounding.learn_new_object)  # TODO replace type
        rospy.Subscriber('MainFind',  OuterObjectEntity, self.find_object_callback)  # TODO replace type
        rospy.Subscriber('MainUpdate', OuterObjectEntity, self.grounding.update_features)  # TODO replace type
        # Wait for messages on topic, go to callback function when new messages arrive.
        rospy.spin()

    def find_object_callback(self, object):
        object = self.grounding.find_object(object)
        self.infopub.publish(object)


if __name__ == '__main__':
    try:
        rospy.init_node('grounding', anonymous=True)
        GroundingNode()
    except rospy.ROSInterruptException:
        pass

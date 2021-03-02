#!/usr/bin/env python3
import rospy
import msgs
from grounding.grounding import Grounding

class GroundingNode():
    def __init__(self):
        self.entitypub = rospy.Publisher('GroundingEntity', msgs.ObjectEntity.msg, queue_size=10)
        self.infopub = rospy.Publisher('GroundingInfo', msgs.ObjectInfo.msg, queue_size=10)
        self.grounding = Grounding()

        while not rospy.is_shutdown():
            self.listener()

    def listener(self):
        rospy.Subscriber('VisionData', int, self.grounding.vision_callback)  # TODO replace type
        rospy.Subscriber('MainLearn', int, self.grounding.learn_new_object)  # TODO replace type
        rospy.Subscriber('MainFind',  msgs.ObjectEntity.msg, self.find_object_callback)  # TODO replace type
        # Wait for messages on topic, go to callback function when new messages arrive.
        rospy.spin()

    def find_object_callback(self, object):
        if object.spatial_desc is None:
            object = self.grounding.find_object(object)
        else:
            object = self.grounding.find_object_with_spatial_desc(object)
        self.infopub.publish(object)



if __name__ == '__main__':
    try:
        GroundingNode()
    except rospy.ROSInterruptException:
        pass

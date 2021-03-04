#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import subprocess

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'I heard %s', )
    subprocess.Popen('gtts-cli "'+data.data+'" | play -t mp3 -',stdin=subprocess.PIPE, shell=True )

def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber('chatter', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()

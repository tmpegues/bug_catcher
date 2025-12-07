"""A node for remapping the turtlesim poses to a usable task space for the bug catcher."""

import rclpy
from rclpy.node import Node
from rclpy.time import Duration

from geometry_msgs.msg import Pose
from turtlesim_msgs.msg import Pose as TPose
from bug_catcher_interfaces.msg import BugInfo
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class TurtleSimAdjuster(Node):
    """Remap turtlesim coordinates to the usable space for the bug catcher."""

    def __init__(self):
        """Initialize the adjuster."""
        super().__init__('turtle_sim_adjuster')

        self.detected = False
        self.turtle_bug = BugInfo(id=0, color='red', target=False)
        self.turtle_bug.pose.pose.orientation.x = 1.0
        self.turtle_bug.pose.pose.orientation.w = 0.0

        self.turtle_sub = self.create_subscription(TPose, 'turtle1/pose', self.pose_callback, 10)
        self.bug_pub = self.create_publisher(BugInfo, 'bug_info', 10)
        self.ee_pub = self.create_publisher(Pose, 'ee_pose', 10)

        self.timer = self.create_timer(1 / 1, self.timer_callback)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def pose_callback(self, pose_msg):
        """
        Remap the incoming pose message.

        Args:
        ----
        pose_msg (turtlesim Pose): the recieved Pose

        """
        # remap the 11 x 11 turtle arena to .5 x .5 centered at 0.5 x 0

        self.turtle_bug.pose.pose.position.x = pose_msg.x / 14
        self.turtle_bug.pose.pose.position.y = pose_msg.y / 11 - 0.5
        self.turtle_bug.pose.pose.position.z = 0.02
        if not self.detected:  # At the first callback, delay 10 seconds before starting to publish
            # BugInfo
            start = self.get_clock().now()
            while (self.get_clock().now() - start) < Duration(seconds=10.0):
                pass
        self.detected = True

    def timer_callback(self):
        """Publish the adjusted turtle pose, if it has been received."""
        if self.detected:
            self.bug_pub.publish(self.turtle_bug)

        # publish the EE pose so I can graph it
        time = rclpy.time.Time()
        try:
            ee_tf = self.tf_buffer.lookup_transform('base', 'fer_hand_tcp', time).transform
            self.get_logger().debug(f'received eetf {ee_tf}')
            ee_pose = Pose()
            ee_pose.position.x = ee_tf.translation.x
            ee_pose.position.y = ee_tf.translation.y
            ee_pose.position.z = ee_tf.translation.z
            ee_pose.orientation = ee_tf.rotation
            self.ee_pub.publish(ee_pose)

        except TransformException:
            pass


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)
    turtle_sim_adjuster = TurtleSimAdjuster()
    rclpy.spin(turtle_sim_adjuster)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

"""The Bug Catcher's main decision making and control node."""

from bug_catcher.bug import Bug
from bug_catcher.bugmover import BugMover
from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher_interfaces.msg import BugsInFrame
import rclpy
from rclpy.node import Node


class CatcherNode(Node):
    """The Bug Catcher's main decision making and control node."""

    def __init__(self):
        """Initialize the Catcher, connecting to the two camera nodes as needed."""
        super().__init__('catcher_node')

        # Initialize the MotionPlanningInterface:
        self.mpi = MotionPlanningInterface(self)

        self.setup_aruco()
        self.setup_bug_color()

        self.bug_list = []
        self.mover = BugMover(self)

        self.get_logger().info('Catcher Node: initialization complete')

    def setup_aruco(self):
        """
        Set up aruco detection for the arena.

        Currently, this will involve setting up listeners to the 'aruco' node, but will eventually
        just initialize the aruco class into this node, or will add a service call to allow this
        node to get aruco poses when specifically requested
        """
        pass

    def setup_bug_color(self):
        """
        Set up color detection for finding bugs in the arena.

        Currently, this involves setting up listeners to the 'bug_color' node, but will eventually
        just initialize the 'bug_color' class into this node.
        """
        self.bugs_in_frame_listener = self.create_subscription(
            BugsInFrame, 'bugs_in_frame', self.bug_list_update, 10
        )
        pass

    def bug_list_update(self, bugs_in_frame):
        """
        Update the persistent list of bugs based on the most recently received frame.

        Args:
        ----
        bugs_in_frame (BugsInFrame): A list of ids and poses for the bugs seen in the last frame

        """
        # Check if any ids already in the list need to be updated with new poses
        for existing_index, existing_bug in enumerate(self.bug_list):
            if existing_bug.ID in bugs_in_frame.id:
                # Get the index of that bug in the new frame
                id_index = bugs_in_frame.id.index(existing_bug.ID)
                # Update the pose
                existing_bug.update(bugs_in_frame.pose[id_index])
                # Remove this bug from the message
                bugs_in_frame.id.pop[id_index]
                bugs_in_frame.pose.pop[id_index]
                bugs_in_frame.color.pop[id_index]

            else:  # Remove the bug if it hasn't been seen in a specific period (1 sec?)
                if existing_bug.pose.header.stamp.sec - self.get_clock().now >= 1:
                    self.bug_list.pop(existing_index)

        # Now that we've checked the bugs that we already know exist, add the rest of the detected
        # bugs to the existing list
        for detected_bug in bugs_in_frame.id:
            self.bug_list.append(Bug(detected_bug.id, detected_bug.pose, detected_bug.color))
        pass

    def decide_which_bug(self):
        """
        Decide which bug to pick.

        Right now, I'm assuming that it will just be base on which bug is closest to the ee.

        Returns
        -------
        bug: the bug closest to the end effector, which should be the easiest bug to pick up.

        """
        min_dist = 100
        best_bug_id = -1
        for existing_bug in self.bug_list:
            dist = self.mover._calc_distance(existing_bug.pose.pose, self.mpi.rs.get_ee_pose())
            if dist < min_dist:
                min_dist = dist
                best_bug_id = existing_bug.ID

        return self.bug_list[best_bug_id]


def main(args=None):
    """Entry point for the Catcher Node."""
    rclpy.init(args=args)
    catcher_node = CatcherNode()
    rclpy.spin(catcher_node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

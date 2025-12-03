"""
The script implements the 'catcher_node' within the 'bug_catcher' package.

Subscribers
-----------
+ /bug_poses (geometry_msgs.msg.PoseArray): Receives lists of detected bug poses
                                            from the vision system.

Services/Actions
----------------
+ /move_group (moveit_msgs.action.MoveGroup): For trajectory planning.
+ /execute_trajectory (moveit_msgs.action.ExecuteTrajectory): For robot movement.

"""

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle

from geometry_msgs.msg import PoseArray, PoseStamped

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from shape_msgs.msg import SolidPrimitive

import tf2_geometry_msgs  # noqa: F401 (Registers PoseStamped for TF2)

from tf2_ros import Buffer, TransformException, TransformListener


class CatcherNode(Node):
    """Bridge the camera frame to the robot's base frame."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('catcher_node')

        self.mpi = MotionPlanningInterface(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.setup_aruco()
        self.setup_bug_color()

        # State flags
        self.is_busy = False
        self.target_bug_name = 'target_bug'
        # Hardcoded size: 5cm x 2cm x 2cm
        self.bug_size = [0.05, 0.02, 0.02]

        self.get_logger().info('Catcher Node: initialization complete. Waiting for bugs...')

    def setup_aruco(self):
        """
        Set up aruco detection for the arena.

        Currently a placeholder. Future implementations will use this to determine
        the robot's location relative to the arena or to identify drop-off zones.
        """
        pass

    def setup_bug_color(self):
        """
        Set up subscribers for the color detection system.

        Uses a MutuallyExclusiveCallbackGroup to ensure the callback does not
        block the main thread or interfere with MoveIt action clients running
        in the MultiThreadedExecutor.
        """
        cb_group = MutuallyExclusiveCallbackGroup()

        # Subscribe to PoseArray from the colordetection node
        self.bugs_in_frame_listener = self.create_subscription(
            PoseArray, '/bug_poses', self.bug_callback, 10, callback_group=cb_group
        )

    def transform_pose(self, input_pose, from_frame, to_frame='base'):
        """
        Transform a pose from the Camera Frame to the Robot Base Frame.

        Args:
        ----
        input_pose (geometry_msgs.msg.Pose): The raw pose from vision.
        from_frame (str): The frame ID of the camera ('camera_color_optical_frame').
        to_frame (str, default='base'): The target frame ID ('base').

        Returns
        -------
        (geometry_msgs.msg.Pose): The transformed pose, or None if TF failed.

        """
        try:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = from_frame
            # Request the transform at the current time
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = input_pose

            # Perform the transform with a 1-second timeout to allow buffer catch-up
            output_pose_stamped = self.tf_buffer.transform(
                pose_stamped, to_frame, timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return output_pose_stamped.pose
        except TransformException as e:
            self.get_logger().warn(f'TF Failed: {e}')
            return None

    def spawn_bug_in_rviz(self, pose):
        """
        Add the detected bug to the MoveIt Planning Scene.

        This creates a collision object in the simulation/planning environment,
        allowing MoveIt to plan grasp trajectories towards it.

        Args:
        ----
        pose (geometry_msgs.msg.Pose): The pose of the bug in the planning frame.

        """
        # Define the shape (Box)
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = self.bug_size

        # Create the Obstacle object
        bug_obstacle = Obstacle(self.target_bug_name, pose, prim)

        # Add to the scene via the MotionPlanningInterface
        self.mpi.ps.add_obstacle(bug_obstacle)
        self.get_logger().info(f'Added {self.target_bug_name} to Planning Scene')

    async def bug_callback(self, msg):
        """
        Handle vision-detected bugs.

        This function executes the core logic pipeline:
        1. Parse vision data.
        2. Transform coordinates to the robot frame.
        3. Update the environment model (spawn object).
        4. Trigger the grasp sequence.

        Args:
        ----
        msg (geometry_msgs.msg.PoseArray): List of bug poses from vision.

        """
        if self.is_busy:
            return
        if len(msg.poses) == 0:
            return

        self.is_busy = True
        self.get_logger().info('Bug detected! Deciding what to do...')

        # === 1. Extract Data ===
        camera_pose = msg.poses[0]
        source_frame = msg.header.frame_id

        # === 2. Transform Coordinate ===
        robot_frame_pose = self.transform_pose(camera_pose, source_frame, to_frame='base')

        if robot_frame_pose is None:
            self.get_logger().error('TF Transform failed. Waiting for TF tree...')
            self.is_busy = False
            return

        # === 3. Spawn in RViz ===
        self.spawn_bug_in_rviz(robot_frame_pose)

        # === 4. Execute Catch Sequence ===
        await self.execute_catch_sequence()

        # Catch multiple bugs continuously
        # self.is_busy = False

    async def execute_catch_sequence(self):
        """
        Execute the physical motion sequence to catch the bug.

        Steps:
        1. Move to Home/Ready.
        2. Open Gripper.
        3. Move to Pre-Grasp (Above object).
        4. Move to Grasp (At object).
        5. Close Gripper.
        6. Lift object.
        """
        bug = self.target_bug_name

        self.get_logger().info('--- Sequence: Get Ready ---')
        if not await self.mpi.GetReady():
            return

        self.get_logger().info('--- Sequence: Open Gripper ---')
        if not await self.mpi.OpenGripper():
            return

        self.get_logger().info(f'--- Sequence: Move Above {bug} ---')
        if not await self.mpi.MoveAboveObject(bug):
            return

        self.get_logger().info(f'--- Sequence: Move Down to {bug} ---')
        if not await self.mpi.MoveDownToObject(bug):
            return

        self.get_logger().info('--- Sequence: Close Gripper (Pick) ---')
        if not await self.mpi.CloseGripper(bug):
            return

        self.get_logger().info('--- Sequence: Lift Up ---')
        if not await self.mpi.LiftOffTable():
            return

        self.get_logger().info('SUCCESS: Bug Caught!')


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)

    catcher_node = CatcherNode()

    executor = MultiThreadedExecutor()
    executor.add_node(catcher_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        catcher_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

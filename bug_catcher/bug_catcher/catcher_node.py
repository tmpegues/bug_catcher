"""
The script implements the 'catcher_node' within the 'bug_catcher' package.

It acts as the physical execution unit. Since the 'color_detection_node' now
handles coordinate transformation (Camera -> Base), this node simply receives
the world-frame coordinates and executes the motion plan.

Subscribers
-----------
+ /wrist_camera/target_bug (bug_catcher_interfaces.msg.BugInfo):
    Receives the target bug pose ALREADY transformed to the Robot Base Frame.

Services/Actions
----------------
+ /move_group (moveit_msgs.action.MoveGroup): For trajectory planning.
+ /execute_trajectory (moveit_msgs.action.ExecuteTrajectory): For robot movement.

Parameters
----------
+ bug_dimensions (double array): [x, y, z] size of the bug for collision model.
+ grasp_height_z (double): The Z height (meters) relative to base to perform the grasp.
+ loop_execution (bool): If true, the node will reset 'is_busy' to catch multiple bugs.

"""

from bug_catcher.motionplanninginterface import MotionPlanningInterface
from bug_catcher.planningscene import Obstacle

from bug_catcher_interfaces.msg import BugInfo


import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from shape_msgs.msg import SolidPrimitive


class CatcherNode(Node):
    """Executes pick-and-place operations based on pre-processed vision data."""

    def __init__(self):
        """Initialize the Catcher, connecting to interfaces."""
        super().__init__('catcher_node')

        # Initialize MotionPlanningInterface
        self.mpi = MotionPlanningInterface(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # =========================================
        # 1. Parameter Declarations
        # =========================================

        # Dimensions of the Hexbug: Length, Width, Height (meters)
        self.declare_parameter('bug_dimensions', [0.045, 0.015, 0.015])
        self.bug_dims = self.get_parameter('bug_dimensions').value

        # The physical height of the table/grasping plane relative to robot base
        # This replaces the hardcoded "0.02"
        self.declare_parameter('grasp_height_z', 0.02)
        self.grasp_z = self.get_parameter('grasp_height_z').value

        # TODO: Continuous execution?
        self.declare_parameter('loop_execution', False)
        self.loop_execution = self.get_parameter('loop_execution').value

        self.declare_parameter('target_bug_name', 'target_bug')
        self.target_bug_name = self.get_parameter('target_bug_name').value

        # =========================================
        # 2. Setup Subscribers
        # =========================================
        self.setup_bug_listener()

        # State flags
        self.is_busy = False

        self.get_logger().info('Catcher Node: Ready. Listening to /wrist_camera/target_bug...')

    def setup_bug_listener(self):
        """Set up subscriber for the Wrist Camera pose data."""
        cb_group = MutuallyExclusiveCallbackGroup()

        self.bugs_in_frame_listener = self.create_subscription(
            BugInfo, '/wrist_camera/target_bug', self.bug_callback, 10, callback_group=cb_group
        )

    def spawn_bug_in_rviz(self, pose):
        """
        Add the detected bug to the MoveIt Planning Scene.

        Args:
        ----
        pose (geometry_msgs.msg.Pose): The pose of the bug in the planning frame.

        """
        # Define the shape (Box) based on parameters
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = self.bug_dims

        # Create the Obstacle object
        bug_obstacle = Obstacle(self.target_bug_name, pose, prim)

        self.mpi.ps.add_obstacle(bug_obstacle)
        self.get_logger().info(f'Added {self.target_bug_name} to Planning Scene')

    async def bug_callback(self, msg):
        """
        Handle incoming target bug from the Wrist Camera.

        Args:
        ----
        msg (bug_catcher_interfaces.msg.BugInfo): The target bug info.

        """
        # Prevent re-entry if the robot is already moving
        if self.is_busy:
            return

        # Valid check for BugInfo (it's a single object, not a list)
        if msg is None:
            return

        self.is_busy = True
        self.get_logger().info('Wrist Camera lock acquired! executing catch...')

        # 1. Extract Pose
        target_pose = msg.pose.pose

        # 2. Apply Z-Height Constraint
        # Vision depth estimation can be noisy. We trust the physical measurement
        # of the table height (parameter) more than the camera's Z estimation.
        target_pose.position.z = self.grasp_z

        self.get_logger().info(
            f'Target Confirmed: x={target_pose.position.x:.2f}, '
            f'y={target_pose.position.y:.2f}, z={target_pose.position.z:.2f}'
        )

        # 3. Spawn in RViz
        self.spawn_bug_in_rviz(target_pose)

        # 4. Execute Catch Sequence
        await self.execute_catch_sequence()

        # 5. Loop Control
        if self.loop_execution:
            self.is_busy = False
            self.get_logger().info('Ready for next target...')
        else:
            self.get_logger().info('Task Complete. Idling.')

    async def execute_catch_sequence(self):
        """Execute the physical motion sequence to catch the bug."""
        bug = self.target_bug_name

        # Ensure RobotState is receiving data
        while rclpy.ok():
            current_joints = self.mpi.rs.get_angles()
            if current_joints is not None and len(current_joints.name) > 0:
                self.get_logger().info('Robot State connected! Starting sequence.')
                break
            await rclpy.sleep(0.1)  # Wait 100ms

        self.get_logger().info('--- Sequence: Get Ready ---')
        if not await self.mpi.GetReady():
            # If planning fails (e.g. out of reach), we abort and reset busy flag
            self.get_logger().error('Failed to reach Ready pose.')
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Open Gripper ---')
        if not await self.mpi.OpenGripper():
            return

        self.get_logger().info(f'--- Sequence: Move Above {bug} ---')
        if not await self.mpi.MoveAboveObject(bug):
            self.get_logger().error('Target out of reach or planning failed.')
            self.is_busy = False
            return

        self.get_logger().info(f'--- Sequence: Move Down to {bug} ---')
        if not await self.mpi.MoveDownToObject(bug):
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Close Gripper (Pick) ---')
        if not await self.mpi.CloseGripper(bug):
            self.is_busy = False
            return

        self.get_logger().info('--- Sequence: Lift Up ---')
        if not await self.mpi.LiftOffTable():
            self.is_busy = False
            return

        self.get_logger().info('SUCCESS: Bug Caught!')


def main(args=None):
    """Run the main entry point for the Catcher Node."""
    rclpy.init(args=args)

    catcher_node = CatcherNode()

    # Use MultiThreadedExecutor to allow MoveIt actions (which use callbacks)
    # to run in parallel with the main node logic.
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

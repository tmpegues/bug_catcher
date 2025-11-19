"""
A python class to enable the user to plan a scene in Rviz.

1) Add or remove boxes to the planning scene dynamically, at any location.
2) Attach and detach collision objects to the robot's end-effector.
3) Load a planning scene from parameters that specify the location and sizes
    of objects.
"""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
)
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
import yaml


class PlanningSceneClass:
    """
    PlanningScene Class to be used by the UserNode.

    PUBLISHERS:
    - /planning_scene (moveit_msgs.msg.PlanningScene)

    """

    def __init__(self, node: Node):
        """Initialize the planning scene."""
        # Define the class variables:
        self.node = node
        self.frame_id = 'base'

        # We define the scene to have no obstacles at initialization.
        self.obstacles = {}
        self.attach_obstacles = {}

        # PUBLISHERS:
        self.planscene = self.node.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )

    def add_obstacle(self, obstacle):
        """Add an individual obstacle to the scene."""
        # Set the object header:
        col_object = CollisionObject()
        col_object.header.frame_id = self.frame_id
        col_object.header.stamp = self.node.get_clock().now().to_msg()

        # Set the object ID:
        col_object.id = obstacle.name

        # Set the Primitive type:
        col_object.primitives = [obstacle.prim]
        col_object.primitive_poses = [obstacle.pose]

        # Add the object to the planning scene:
        col_object.operation = CollisionObject.ADD

        # Update the Planning Scene:
        pscene = PlanningScene()
        pscene.world.collision_objects = [col_object]
        pscene.is_diff = True

        # Publish the Planning Scene:
        self.planscene.publish(pscene)

        # Add the object to the list:
        self.obstacles[obstacle.name] = obstacle

    def remove_obstacle(self, obstacle):
        """Remove an individual obstacle from the scene."""
        # Set the object header:
        col_object = CollisionObject()
        col_object.header.frame_id = self.frame_id
        col_object.header.stamp = self.node.get_clock().now().to_msg()
        # Set the object ID:
        col_object.id = obstacle.name

        # Remove the object from the planning scene:
        col_object.operation = CollisionObject.REMOVE

        # Update the Planning Scene:
        pscene = PlanningScene()
        pscene.world.collision_objects = [col_object]
        pscene.is_diff = True

        # Publish the Planning Scene:
        self.planscene.publish(pscene)

        # Remove the object from the list:
        self.obstacles.pop(obstacle.name, None)

    def attach_obstacle(self, name):
        """Attach the object to the gripper."""
        obstacle = self.obstacles[name]

        # Add the obastacle to the AttachedCollisionObject:
        att_object = AttachedCollisionObject()
        # Attach to one of the fingers
        att_object.link_name = 'fer_hand_tcp'
        # Define the object parameters:
        att_object.object.header.frame_id = self.frame_id
        att_object.object.header.stamp = self.node.get_clock().now().to_msg()
        att_object.object.id = obstacle.name
        att_object.object.primitives = [obstacle.prim]
        att_object.object.primitive_poses = [obstacle.pose]
        att_object.touch_links = ['fer_gripper_left_finger', 'fer_gripper_right_finger']
        # Set the operation to add for the attached object
        att_object.object.operation = CollisionObject.ADD

        # Update the robot state:
        pscene = PlanningScene()
        pscene.robot_state.is_diff = True
        pscene.robot_state.attached_collision_objects = [att_object]
        pscene.is_diff = True

        # Publish the Planning Scene:
        self.planscene.publish(pscene)

        # Add the object to the attached object list:
        self.attach_obstacles[name] = self.obstacles[name]

    def detach_obstacle(self, name):
        """Remove the object from the gripper."""
        obstacle = self.attach_obstacles[name]

        det_object = AttachedCollisionObject()
        det_object.link_name = 'fer_hand_tcp'
        # Define the object parameters
        det_object.object.header.frame_id = self.frame_id
        det_object.object.header.stamp = self.node.get_clock().now().to_msg()
        det_object.object.id = name
        det_object.object.id = obstacle.name
        # Set the operation to remove the attached object
        det_object.object.operation = det_object.object.REMOVE

        # Update the robot state to remove the object:
        pscene = PlanningScene()
        pscene.robot_state.is_diff = True
        pscene.robot_state.attached_collision_objects = [det_object]
        pscene.is_diff = True

        # Publish the scene:
        self.planscene.publish(pscene)

        # Remove the obstacle from the attached object list:
        self.attach_obstacles.pop(obstacle.name, None)

    def load_scene(self, filename):
        """
        Load the obstacles into the planning_scene.

        Each obstacle in the file will be created using the Obstacle class.
        The data will be comma seperated in the order of name, location(x,y,z),
        orientation(x,y,z,w), then size(type,dimension(x,y,z)).
        """
        # Read through the yaml file and create Obstacles():
        pkg_share = get_package_share_directory('bug_catcher')
        file = Path(pkg_share) / 'config' / filename
        with open(file, 'r') as f:
            data = yaml.safe_load(f)

        for entry in data:
            # Construct the pose of the object
            pose = Pose()
            pose.position.x = entry['pose']['position']['x']
            pose.position.y = entry['pose']['position']['y']
            pose.position.z = entry['pose']['position']['z']
            pose.orientation.x = entry['pose']['orientation']['x']
            pose.orientation.y = entry['pose']['orientation']['y']
            pose.orientation.z = entry['pose']['orientation']['z']
            pose.orientation.w = entry['pose']['orientation']['w']

            # Construct SolidPrimitive type
            prim = SolidPrimitive()
            prim.type = entry['size']['type']
            prim.dimensions = entry['size']['dimensions']

            # Create the Obstacle and add it to the world:
            obstacle = Obstacle(entry['name'], pose, prim)
            self.add_obstacle(obstacle)


class Obstacle:

    def __init__(self, name, pose, prim):
        """
        Initialize an obstacle.

        Argument(s)
        ----------
        name: The name of the object.
        pose: The location of the object in a frame: Pose (pose,orientation).
        prim: The size of the object: SolidPrimitive (type,dimension).

        """
        # Define the class variables:
        self.name = name
        self.pose = pose
        self.prim = prim

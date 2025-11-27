"""Helper functions for the BugMover class."""

import math

from geometry_msgs.msg import Point, Pose


def _calc_distance(p1: Point | Pose, p2: Point | Pose):
    """
    Calculate the Euclidean distance between two points or poses.

    Args:
    ----
    p1 (Point | Pose): the initial position to calculate distance from
    p2 (Point | Pose): the final position to calculate distance to

    Returns
    -------
    (float) the Euclidian distance between the Points or Poses

    """
    if type(p1) is Pose:
        p1 = p1.position
    if type(p2) is Pose:
        p2 = p2.position
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def _calc_travel_time(start_pose: Pose, end_pose: Pose) -> float:
    """Calculate the estimated travel time needed for robot to move from start to end."""
    distance = _calc_distance(start_pose.position, end_pose.position)

    ROBOT_SPEED = 0.1  # TODO: Update with actual robot speed
    OVERHEAD_TIME = 1.0  # TODO: Update with actual overhead time

    travel_time = distance / ROBOT_SPEED + OVERHEAD_TIME
    return travel_time


def _distance_scaler(p1: Pose, p2: Pose, max_step: float = 0.1):
    """
    Calculate the Pose that is the specified distance from p1 to p2.

    The orientation is not calculated, it's taken directly from p2.

    Args:
    p1 (Pose): The pose you want to step from.
    p2 (Pose): The pose you want to step towards.
    max_step (float): Maximum step size (in meters) to take from p1 to p2. Defaults to 0.1 m.

    Returns
    -------
    step_pose (Pose): The pose that is a single step from p1 to p2, with the orientation of p2

    """
    # Check distance and scale the step if dist > max_step
    dist_to_bug = _calc_distance(p2, p1)
    if dist_to_bug <= max_step:
        step_pose = p2  # TMP TODO: Check how to copy the pose
    else:
        step_pose = Pose(orientation=p2.orientation)
        p1_vec = [p1.position.x, p1.position.y, p1.position.z]
        p2_vec = [p2.position.x, p2.position.y, p2.position.z]
        step_vec = [(x2 - x1) / dist_to_bug * max_step for x1, x2 in zip(p1_vec, p2_vec)]
        step_point = Point(x=step_vec[0], y=step_vec[1], z=step_vec[2])
        step_pose.position = step_point
    return step_pose

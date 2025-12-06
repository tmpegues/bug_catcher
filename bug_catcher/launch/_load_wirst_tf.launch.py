import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription

from launch_ros.actions import Node

import yaml


def generate_launch_description():
    """Launch wrist to camera static transform publisher node."""
    pkg_share = get_package_share_directory('bug_catcher')
    yaml_file_path = os.path.join(pkg_share, 'config', 'wrist_cam_calibration.yaml')
    try:
        with open(yaml_file_path, 'r') as f:
            calib_data = yaml.safe_load(f)
        trans = calib_data['transform']['translation']
        rot = calib_data['transform']['rotation']

        x = str(trans['x'])
        y = str(trans['y'])
        z = str(trans['z'])
        qx = str(rot['x'])
        qy = str(rot['y'])
        qz = str(rot['z'])
        qw = str(rot['w'])

        print(f'Loaded Wrist TF from YAML: x={x}, y={y}, z={z}')

    except (FileNotFoundError, yaml.YAMLError, KeyError) as e:
        print(f'Error loading wrist calibration: {e}')
        x, y, z = '0.05', '0.0', '0.05'
        qx, qy, qz, qw = '0', '0', '0', '1'

    return LaunchDescription(
        [
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='wrist_cam_tf_loader',
                arguments=[x, y, z, qx, qy, qz, qw, 'fer_hand_tcp', 'wrist_camera_link'],
            )
        ]
    )

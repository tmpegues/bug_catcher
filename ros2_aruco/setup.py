from pathlib import Path

from setuptools import find_packages, setup


def recursive_files(prefix, path):
    """
    Recurse over path returning a list of tuples suitable for use with setuptools data_files.

    :param prefix: prefix path to prepend to the path
    :param path: Path to directory to recurse. Path should not have a trailing '/'
    :return: List of tuples. 1st element of each tuple is destination path, 2nd element is a list
             of files to copy to that path
    """
    return [(str(Path(prefix)/subdir),
            [str(file) for file in subdir.glob('*') if not file.is_dir()])
            for subdir in Path(path).glob('**')]


package_name = 'ros2_aruco'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *recursive_files('share/' + package_name, 'launch'),
        *recursive_files('share/' + package_name, 'config'),
        *recursive_files('share/' + package_name, 'models'),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nolan Knight',
    maintainer_email='nolanknight2026@u.northwestern.edu',
    description='This package creates a tracking system for our bug project.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'aruco_node = ros2_aruco.aruco_node:main',
            'aruco_generate_marker = ros2_aruco.aruco_generate_marker:main',
            'calibration_node = ros2_aruco.calibration_node:main',
        ],
    },
)

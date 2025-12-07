"""Package setup."""

from pathlib import Path

from setuptools import find_packages, setup


def recursive_files(prefix, path):
    """
    Recursive path list of tuples suitable for use with setuptools data_files.

    :param prefix: prefix path to prepend to the path
    :param path: Path to directory to recurse. Path should not have a trailing
                 '/'
    :return: List of tuples. First element of each tuple is destination path,
                second element is a list of files to copy to that path.
    """
    return [
        (str(Path(prefix) / subdir), [str(file) for file in subdir.glob('*') if not file.is_dir()])
        for subdir in Path(path).glob('**')
    ]


package_name = 'bug_catcher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *recursive_files('share/' + package_name, 'launch'),
        *recursive_files('share/' + package_name, 'config'),
    ],
    install_requires=[
        'setuptools',
        'scikit-image',
        'numpy',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='Miguel Pegues',
    maintainer_email='MiguelP@u.northwestern.edu',
    description='This package creates a MotionPlanningInterface for the FER to pick up and \
        retrieve an object.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'catcher_node = bug_catcher.catcher_node:main',
            'pick_node = bug_catcher.pick_node:main',
            'calibration_node = bug_catcher.Calibration_TargetPublisher_April:main',
            'target_decision_node = bug_catcher.TargetDecision:main',
            'turtle_sim_adjuster = bug_catcher.turtle_sim_adjuster:main',
        ],
    },
)

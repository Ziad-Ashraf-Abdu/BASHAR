from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'bashar'

setup(
    name=package_name,
    version='1.0.0',
    # Automatically find the bashar package inside the src/ folder
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Tell ROS 2 where to find the config profiles if we ship defaults
        (os.path.join('share', package_name, 'config', 'profiles'), glob('config/profiles/*.json')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Ziad Ashraf',
    maintainer_email='ziad.mohamed04@eng-st.cu.edu.eg',
    description='BASHAR: Hardware-agnostic Kinematics, Dynamics, and Control Library',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
    },
)
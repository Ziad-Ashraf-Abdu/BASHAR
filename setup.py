from setuptools import setup, find_packages
import os

package_name = 'bashar'

# Read the README.md for the PyPI long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bashar', 
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.20.0',
    ],
    author='Zeyad Ashraf',
    author_email='ziad.mohamed04@eng-st.cu.edu.eg',
    description='Hardware-agnostic Kinematics, Dynamics, and Control Middleware',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ziad-ashraf-abdu/bashar',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)
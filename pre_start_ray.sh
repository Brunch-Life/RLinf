#!/bin/bash
export CUDA_VISIBLE_DEVICES=""

unset PYTHONPATH
export PYTHONPATH="/home/ubuntu/RLinf"

# Modify these environment variables as needed
# export RLINF_NODE_RANK=3 # Change this to the appropriate node rank if using multiple nodes
# export RLINF_NODE_RANK=2
export RLINF_NODE_RANK=6
# export RLINF_NODE_RANK=4
export RLINF_COMM_NET_DEVICES="rlinf" # Change this if you use a different network interface

# export RLINF_NODE_RANK=1 # Change this to the appropriate node rank if using multiple nodes
# export RLINF_COMM_NET_DEVICES="wlp0s20f3" # Change this if you use a different network interface

# If you are using the docker image, change this to source switch_env franka-<version>, e.g., switch_env franka-0.15.0
source .venv/bin/activate # Source your virtual environment here
source /home/ubuntu/catkin_franka/devel/setup.bash
# source /opt/ros/noetic/setup.bash

# Additionally source your own catkin workspace setup.bash if you are not installing franka_ros and serl_franka_controllers via the docker image or installation script
# source <your_catkin_ws>/devel/setup.bash

# ray start --address=10.126.126.2:6379
# ray start --address='10.126.126.100:6379'
# ray start --address='192.168.110.41:6379'
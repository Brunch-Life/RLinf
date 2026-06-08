#!/bin/bash

set -eo pipefail

# Install the ROS Noetic runtime pieces needed by the XSquare Turtle2 Python
# controller. Turtle2-specific message packages are expected to come from the
# robot workspace or xsquare_turtle_basics; this script installs the common ROS
# stack that is safe to provide in the RLinf image.

if ! command -v apt-get &> /dev/null; then
    echo "apt-get could not be found. This script is intended for Debian-based systems."
    exit 1
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

if ! sudo -n true 2>/dev/null; then
    if [ "$EUID" -eq 0 ]; then
        apt-get update -y
        apt-get install -y --no-install-recommends sudo
    else
        echo "This script requires sudo privileges. Please run as a user with sudo access."
        exit 1
    fi
fi

sudo apt-get -o Acquire::Retries=5 update -y
sudo apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    wget \
    curl \
    lsb-release \
    gnupg \
    cmake \
    build-essential

ubuntu_codename=""
if command -v lsb_release >/dev/null 2>&1; then
    ubuntu_codename=$(lsb_release -cs || true)
elif [ -f /etc/os-release ]; then
    ubuntu_codename=$(grep '^UBUNTU_CODENAME=' /etc/os-release | cut -d= -f2)
fi

if [ "$ubuntu_codename" != "focal" ]; then
    echo "ROS Noetic Turtle2 image must be built on Ubuntu 20.04 (focal); got '$ubuntu_codename'." >&2
    exit 1
fi

ros_mirror="http://mirrors.ustc.edu.cn/ros/ubuntu"
source_line="deb ${ros_mirror} ${ubuntu_codename} main"

if sudo grep -Rqs -- "$source_line" /etc/apt/sources.list /etc/apt/sources.list.d 2>/dev/null; then
    echo "ROS source already present in /etc/apt, skipping addition: $source_line"
else
    echo "$source_line" | sudo tee /etc/apt/sources.list.d/ros-latest.list >/dev/null
    echo "Added ROS source: $source_line"
fi

sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

sudo apt-get -o Acquire::Retries=5 update -y
sudo apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-std-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-nav-msgs

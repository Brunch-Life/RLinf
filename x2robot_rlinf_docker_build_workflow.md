# X2Robot RLinf Docker Image Build Workflow

This document records the current build workflow for the RLinf Turtle2/X2Robot
environment image.

The image is for the RLinf-side runtime and Ray worker environment. It is not
the original Turtle2 controller image. The original controller container, such
as `turtle2_release`, should stay unchanged.

## Current Image

Current local image:

```text
rlinf:embodied-xsquare_turtle2-test
```

Observed metadata:

```text
image id: sha256:46bf5e5a115362849ed5478ff615f1afe8954288a270fb4e20fbb32571a220df
created: 2026-06-04T14:26:40.524541583+08:00
arch: amd64
os: linux
docker images size: about 16.9GB
```

The image installs dependencies and the `xsquare_turtle2` virtualenv. It does
not install the RLinf source tree into the image. Runtime containers should
mount the RLinf source directory to `/workspace/RLinf` and set `PYTHONPATH`.

## Source Files

The current image is built from:

```text
docker/Dockerfile
requirements/install.sh
requirements/embodied/ros_turtle2_install.sh
```

Important Dockerfile path:

```text
BUILD_TARGET=embodied-xsquare_turtle2
target stage=embodied-xsquare_turtle2-image
base image=ubuntu:20.04
```

The Turtle2 stage runs:

```bash
bash requirements/install.sh ${INSTALL_MIRROR_OPTION} \
  --platform nvidia \
  embodied \
  --venv xsquare_turtle2 \
  --env xsquare_turtle2
```

That install path:

- runs `uv sync --extra xsquare_turtle2`
- installs ROS Noetic runtime packages through
  `requirements/embodied/ros_turtle2_install.sh`
- installs `RLinf/xsquare_turtle_basics`
- appends ROS setup scripts to `/opt/venv/xsquare_turtle2/bin/activate`

## Build On A GPU Head Machine

Run from the RLinf repository root:

```bash
cd /path/to/RLinf

sudo docker build \
  -f docker/Dockerfile \
  --target embodied-xsquare_turtle2-image \
  --build-arg BUILD_TARGET=embodied-xsquare_turtle2 \
  --build-arg PLATFORM=nvidia \
  -t rlinf:embodied-xsquare_turtle2-test \
  .
```

If GitHub access needs a proxy, preserve the proxy variables for `docker build`.
APT inside `ros_turtle2_install.sh` intentionally unsets proxies and uses the
configured mirrors.

```bash
sudo --preserve-env=HTTP_PROXY,HTTPS_PROXY,ALL_PROXY,NO_PROXY \
  docker build \
    -f docker/Dockerfile \
    --target embodied-xsquare_turtle2-image \
    --build-arg BUILD_TARGET=embodied-xsquare_turtle2 \
    --build-arg PLATFORM=nvidia \
    -t rlinf:embodied-xsquare_turtle2-test \
    .
```

If a previous partial build polluted the cache, rebuild with:

```bash
sudo docker build --no-cache \
  -f docker/Dockerfile \
  --target embodied-xsquare_turtle2-image \
  --build-arg BUILD_TARGET=embodied-xsquare_turtle2 \
  --build-arg PLATFORM=nvidia \
  -t rlinf:embodied-xsquare_turtle2-test \
  .
```

## Verify The Image

Check image metadata:

```bash
sudo docker images rlinf:embodied-xsquare_turtle2-test
sudo docker inspect rlinf:embodied-xsquare_turtle2-test \
  --format 'Id={{.Id}} Created={{.Created}} Arch={{.Architecture}} Os={{.Os}} Size={{.Size}}'
```

Run a local import check:

```bash
sudo docker run --rm \
  --network host \
  --ipc host \
  -v /path/to/RLinf:/workspace/RLinf \
  -w /workspace/RLinf \
  rlinf:embodied-xsquare_turtle2-test \
  bash -lc '
    source /opt/venv/xsquare_turtle2/bin/activate
    export PYTHONPATH=/workspace/RLinf:${PYTHONPATH}
    python -c "import rlinf; import rospy; import cv_bridge; import turtle2_basic; print(\"import ok\")"
  '
```

The local GPU/head machine normally will not have the Turtle2 runtime message
workspace, so imports such as `arm_control` are expected to work only in the
remote Turtle2 worker container where the Turtle2 workspace is mounted and
sourced.

## Save And Transfer The Image

Use this when the target machine should not rebuild the image:

```bash
sudo docker save rlinf:embodied-xsquare_turtle2-test \
  | zstd -T0 -19 -o /tmp/rlinf-xsquare_turtle2.tar.zst

rsync -P /tmp/rlinf-xsquare_turtle2.tar.zst <target>:/tmp/
```

On the target machine:

```bash
zstd -dc /tmp/rlinf-xsquare_turtle2.tar.zst | sudo docker load
sudo docker images rlinf:embodied-xsquare_turtle2-test
```

## Run As Remote Turtle2 Worker

The runtime container must mount both the RLinf source tree and the Turtle2
runtime workspace.

```bash
export RLINF_IMAGE=rlinf:embodied-xsquare_turtle2-test
export REMOTE_RLINF=/home/arm/RLinf_feature_turtle2_deploy
export TURTLE_RUNTIME=/home/arm/rlinf_turtle2_runtime
export RAY_HEAD_IP=<gpu_head_ip>

docker rm -f rlinf_turtle2_worker 2>/dev/null || true

docker run -d \
  --name rlinf_turtle2_worker \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --shm-size 16g \
  -v ${REMOTE_RLINF}:/workspace/RLinf \
  -v ${TURTLE_RUNTIME}/modules/devel:/home/arm/prj/turtle2/modules/devel:ro \
  -v ${TURTLE_RUNTIME}/modules/src/trajectory_smooth:/home/arm/prj/turtle2/modules/src/trajectory_smooth:ro \
  -e RLINF_NODE_RANK=1 \
  -e RLINF_COMM_NET_DEVICES=enp1s0 \
  -e RAY_HEAD_IP=${RAY_HEAD_IP} \
  -w /workspace/RLinf \
  ${RLINF_IMAGE} \
  bash -lc 'set -e
    source /opt/venv/xsquare_turtle2/bin/activate
    source /opt/ros/noetic/setup.bash
    source /home/arm/prj/turtle2/modules/devel/setup.bash
    export PYTHONPATH=/workspace/RLinf:${PYTHONPATH}
    ray stop -f || true
    ray start --address=${RAY_HEAD_IP}:6379 --node-ip-address=192.168.120.155 --disable-usage-stats --block'
```

## Correct Deployment Topology

For the intended deployment:

- GPU/head machine runs Ray head and the eval driver natively from the project
  `.venv`.
- Turtle2 slave machine runs the RLinf Docker image as a Ray worker.
- Turtle2 slave machine keeps the original controller Docker unchanged.

The local `rlinf_head` Docker container was only a temporary smoke-test head.
It should not be treated as the final deployment topology.

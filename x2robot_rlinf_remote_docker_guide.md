# X2Robot RLinf 远端 Docker 启动与运行指南

本文档说明如何在 `sohu-turtle-1` 上启动新建的 RLinf Docker，并让它作为
Ray worker 承载 Turtle2/X2Robot 的真机 env 组件。原有控制器 Docker 不需要修改。

当前已验证的远端状态：

- 远端别名：`sohu-turtle-1`
- 远端主机 IP：`192.168.120.155`
- 已上传镜像：`rlinf:embodied-xsquare_turtle2-test`
- 镜像 ID：`46bf5e5a1153`
- 原控制器容器：`turtle2_release`，镜像 `quantum1:v1`，`host` 网络
- Turtle2 runtime workspace：`/home/arm/rlinf_turtle2_runtime`

## 1. 先确认远端基础状态

在本机执行：

```bash
ssh sohu-turtle-1
```

在远端执行：

```bash
docker images rlinf:embodied-xsquare_turtle2-test
docker ps --format '{{.Names}} {{.Image}} {{.Status}} {{.Networks}}'
ls -ld /home/arm/rlinf_turtle2_runtime/modules/devel
ls -ld /home/arm/rlinf_turtle2_runtime/modules/src/trajectory_smooth
ip -br addr
```

必须满足：

- `rlinf:embodied-xsquare_turtle2-test` 存在。
- `turtle2_release quantum1:v1` 正在运行，网络是 `host`。
- `/home/arm/rlinf_turtle2_runtime/modules/devel/setup.bash` 存在。
- 远端用于和 head 节点通信的网卡当前是 `enp1s0`，IP 是 `192.168.120.155`。

## 2. 确认远端 RLinf 代码目录

镜像里只安装了依赖和 `xsquare_turtle2` venv，没有安装 RLinf 项目本身。
因此运行程序时必须挂载一份远端 RLinf 代码目录，并设置 `PYTHONPATH`。

先确认远端哪个目录是你要跑的代码：

```bash
for d in /home/arm/RLinf /home/arm/cynws/RLinf /home/arm/Jiahao/RLinf; do
  if [ -d "$d/.git" ]; then
    echo "==== $d"
    git -C "$d" branch --show-current
    git -C "$d" rev-parse --short HEAD
    git -C "$d" status --short | head
  fi
done
```

如果远端代码不是当前开发分支，需要先同步代码。示例：

```bash
rsync -az \
  --exclude .git \
  --exclude .venv \
  --exclude logs \
  --exclude results \
  --exclude __pycache__ \
  ./ sohu-turtle-1:/home/arm/RLinf_feature_turtle2_deploy/
```

后续命令假设使用：

```bash
export REMOTE_RLINF=/home/arm/RLinf_feature_turtle2_deploy
```

如果你使用已有目录，把 `REMOTE_RLINF` 改成对应路径。

## 3. 进入 RLinf Docker 做手动检查

`turtle2_basic` 内部硬编码读取：

```text
/home/arm/prj/turtle2/modules/devel
/home/arm/prj/turtle2/modules/src/trajectory_smooth
```

所以启动容器时要把 runtime workspace 挂载到这两个路径。

在远端执行：

```bash
export RLINF_IMAGE=rlinf:embodied-xsquare_turtle2-test
export TURTLE_RUNTIME=/home/arm/rlinf_turtle2_runtime
export REMOTE_RLINF=/home/arm/RLinf_feature_turtle2_deploy

docker run --rm -it \
  --name rlinf_turtle2_shell \
  --network host \
  --ipc host \
  --shm-size 16g \
  -v ${REMOTE_RLINF}:/workspace/RLinf \
  -v ${TURTLE_RUNTIME}/modules/devel:/home/arm/prj/turtle2/modules/devel:ro \
  -v ${TURTLE_RUNTIME}/modules/src/trajectory_smooth:/home/arm/prj/turtle2/modules/src/trajectory_smooth:ro \
  -w /workspace/RLinf \
  ${RLINF_IMAGE} \
  bash
```

进入容器后执行：

```bash
source /opt/venv/xsquare_turtle2/bin/activate
source /opt/ros/noetic/setup.bash
source /home/arm/prj/turtle2/modules/devel/setup.bash
export PYTHONPATH=/workspace/RLinf:${PYTHONPATH}
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp1s0
```

最小导入检查：

```bash
python -c "import rlinf; import rospy; import cv_bridge; import arm_control; import turtle2_msgs_srvs; import chassis_control_center; import turtle2_basic; print('import ok')"
```

检查是否能看到原控制器 Docker 的 ROS topics：

```bash
timeout 10 rostopic list | egrep '/follow_pos_cmd_1|/follow_pos_cmd_2|/joint_information|/camera[123]/usb_cam[123]/image_raw|/chassis/cmd_vel|/head/control'
```

只做 controller 初始化检查，不发控制动作：

```bash
python -c "import rospy; rospy.init_node('rlinf_turtle2_import_test', anonymous=True, disable_signals=True); from turtle2_basic.turtle2_controller.Turtle2Controller import Turtle2Controller; c=Turtle2Controller(init_node=False); print('controller init ok', type(c.head).__name__, type(c.arms).__name__, type(c.chassis).__name__)"
```

## 4. 启动远端 RLinf Ray worker 容器

真实跑 `realworld_x2robot_eval` 时，入口脚本在 GPU/head 节点执行；远端
`sohu-turtle-1` 上这个容器只负责 Ray worker 和真机 env。

先在 GPU/head 节点启动 Ray head。示例：

```bash
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<gpu_head_network_interface>

ray stop -f || true
ray start --head --port=6379 --node-ip-address=<gpu_head_ip>
```

然后在 `sohu-turtle-1` 启动 worker 容器：

```bash
export RLINF_IMAGE=rlinf:embodied-xsquare_turtle2-test
export TURTLE_RUNTIME=/home/arm/rlinf_turtle2_runtime
export REMOTE_RLINF=/home/arm/RLinf_feature_turtle2_deploy
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
    ray start --address=${RAY_HEAD_IP}:6379 --node-ip-address=192.168.120.155 --block'
```

检查 worker 是否连上：

```bash
docker logs -f rlinf_turtle2_worker
```

在 GPU/head 节点执行：

```bash
ray status
```

应能看到两个节点：rank 0 的 GPU/head 节点，以及 rank 1 的 `sohu-turtle-1`。

## 5. 在 head 节点运行 RLinf 程序

当前 `examples/embodiment/config/realworld_x2robot_eval.yaml` 的拓扑是：

- `actor` 和 `rollout` 放在 node group `"4090"`，即 node rank `0`
- `env` 放在 node group `turtle2`，即 node rank `1`
- `cluster.num_nodes: 2`

运行入口应在 GPU/head 节点执行，而不是在 `sohu-turtle-1` 执行。

先做 dummy 链路测试，不让机械臂动：

```bash
source <head_node_python_env>/bin/activate
export PYTHONPATH=/path/to/RLinf:${PYTHONPATH}
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=<gpu_head_network_interface>

bash examples/embodiment/run_realworld_eval.sh realworld_x2robot_eval \
  env.train.override_cfg.is_dummy=True \
  env.eval.override_cfg.is_dummy=True \
  runner.ckpt_path=/path/to/x2robot_ckpt.pt \
  rollout.model.model_path=/path/to/x2robot_model
```

dummy 通过后，再改成真机：

```bash
bash examples/embodiment/run_realworld_eval.sh realworld_x2robot_eval \
  env.train.override_cfg.is_dummy=False \
  env.eval.override_cfg.is_dummy=False \
  runner.ckpt_path=/path/to/x2robot_ckpt.pt \
  rollout.model.model_path=/path/to/x2robot_model
```

真机运行前必须确认 `examples/embodiment/config/env/realworld_x2robot.yaml` 中这些值已经按实机标定更新：

- `reset_ee_pose`
- `target_ee_pose`
- `ee_pose_limit_min`
- `ee_pose_limit_max`
- `gripper_width_limit_min`
- `gripper_width_limit_max`
- `camera_names`
- `use_camera_ids`
- `use_arm`

## 6. 运行期间常用命令

查看远端 worker 容器日志：

```bash
ssh sohu-turtle-1 "docker logs -f rlinf_turtle2_worker"
```

进入远端 worker 容器：

```bash
ssh sohu-turtle-1 "docker exec -it rlinf_turtle2_worker bash"
```

停止远端 worker 容器：

```bash
ssh sohu-turtle-1 "docker rm -f rlinf_turtle2_worker"
```

检查原控制器容器是否还在：

```bash
ssh sohu-turtle-1 "docker ps --format '{{.Names}} {{.Image}} {{.Status}} {{.Networks}}' | grep turtle2_release"
```

在远端 RLinf 容器里检查 ROS topic：

```bash
docker exec -it rlinf_turtle2_worker bash -lc '
  source /opt/venv/xsquare_turtle2/bin/activate
  source /opt/ros/noetic/setup.bash
  source /home/arm/prj/turtle2/modules/devel/setup.bash
  rostopic list | egrep "/follow_pos_cmd_1|/follow_pos_cmd_2|/joint_information|/camera[123]/usb_cam[123]/image_raw|/chassis/cmd_vel|/head/control"
'
```

## 7. 关键注意事项

- 必须使用 `--network host`，否则新 RLinf 容器无法稳定连接原控制器 Docker 的 ROS master/topics。
- 不要修改原 `turtle2_release` 控制器容器；RLinf 只通过 ROS topic/service 接入。
- `RLINF_NODE_RANK=1` 必须在远端 `ray start` 前设置。
- `RLINF_NODE_RANK=0` 必须在 head 节点 `ray start --head` 前设置。
- 远端容器挂载路径必须保持为 `/home/arm/prj/turtle2/modules/...`，这是 `turtle2_basic` 当前代码期望的路径。
- 镜像不包含 RLinf 源码；每次换分支或改代码后，都要保证远端挂载的 `REMOTE_RLINF` 是最新代码。
- 真机运行前先跑 dummy，再跑真实 `is_dummy=False`。

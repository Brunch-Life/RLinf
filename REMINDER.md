# REMINDER

## `RLinf/xsquare_turtle_basics` has a Turtle2 chassis virtual-zero bug

Observed while testing `toolkits/realworld_check/test_turtle2_controller.py` on
`sohu-turtle-1`.

Source package:

- Repository: `https://github.com/RLinf/xsquare_turtle_basics.git`
- Installed package: `turtle2_basic`
- Runtime file:
  `/opt/venv/xsquare_turtle2/lib/python3.11/site-packages/turtle2_basic/turtle2_controller/controllers.py`

Current trigger in this repo:

- `rlinf/envs/realworld/xsquare/turtle2_smooth_controller.py`
- `Turtle2SmoothController.__init__()` calls
  `self.controller.chassis_set_current_pose_as_virtual_zero()`.

Problem in `turtle2_basic`:

```python
def set_virtual_zero(self, pose):
    if pose is not list or len(pose) != 7:
        rospy.logerr("pose must be a list of length 7")
        rospy.logerr(f"Invalid pose: {pose}")
        return
```

`pose is not list` is a type-object identity check, so it rejects normal list
instances. This logs an error even when `pose` is a valid 7-element pose.

There is also a similar type check later:

```python
if self.virtual_zero_tf is not np.ndarray:
```

Expected fix in `RLinf/xsquare_turtle_basics`:

```python
if not isinstance(pose, list) or len(pose) != 7:
    ...

if not isinstance(self.virtual_zero_tf, np.ndarray):
    ...
```

Current impact:

- Arm/head/lift `get_state()` worked during testing.
- ROS topics were visible from the RLinf Docker.
- The error appears during Turtle2 controller initialization and affects chassis
  virtual-zero / relative chassis pose logic.
- Leave it unfixed in this RLinf repo for now; fix it upstream in
  `RLinf/xsquare_turtle_basics` later.

## `test_turtle2_controller.py` is not directly usable in the current two-node setup

Observed while testing `go` and `reset` on `sohu-turtle-1`.

File:

- `toolkits/realworld_check/test_turtle2_controller.py`

Problems:

1. It launches the controller with the default node rank:

```python
controller = Turtle2SmoothController.launch_controller(freq=50)
```

`Turtle2SmoothController.launch_controller()` defaults to `node_rank=0`, so in
the current two-node Ray setup this schedules the controller on the local
GPU/head node instead of the Turtle2 robot node. For the current deployment it
must explicitly use `node_rank=1`.

2. The `go` command calls a method that does not exist in the current
`Turtle2SmoothController`:

```python
controller.move_arm(...).wait()
```

Current available methods are:

- `move_delta(left_arm_target, right_arm_target)`
- `move_abs(left_arm_target, right_arm_target)`
- `reset_arms()`

What worked during testing:

```python
controller = Turtle2SmoothController.launch_controller(freq=50, node_rank=1)
controller.move_delta(
    [0, 0, 0, 0, 0, 0, 0],
    [0.27, 0.09, 0.06, 0.0, 1.0, 0.5, 0.0],
).wait()
```

Then reset:

```python
controller.reset_arms().wait()
```

Observed result:

- `go` moved the right arm.
- `reset` returned the right arm to near zero.
- No source change was made yet; update this test script later if it should be
  used as the standard Turtle2 smoke test.

## Local Ray head container cannot directly import Turtle2 controller messages

Observed while running a no-motion controller smoke test from the local
`rlinf_head` container.

Command shape:

```python
from rlinf.envs.realworld.xsquare.turtle2_smooth_controller import (
    Turtle2SmoothController,
)
```

Failure:

```text
ModuleNotFoundError: No module named 'arm_control'
```

Reason:

- `turtle2_basic` imports ROS message packages such as `arm_control`.
- Those message packages come from the Turtle2 runtime workspace:
  `/home/arm/prj/turtle2/modules/devel/setup.bash`.
- That workspace is mounted and sourced in the remote `sohu-turtle-1`
  `rlinf_turtle2_worker` container, but not in the local GPU/head
  `rlinf_head` container.

Current workaround:

- Run Turtle2 controller smoke tests from the remote worker container, after
  sourcing both ROS and Turtle2 workspace setup files.
- In normal RLinf eval, keep the env/controller component placed on
  `node_rank=1`, so the actual actor imports and runs on the Turtle2 node.

Observed remote no-motion check:

- `Turtle2SmoothController.launch_controller(freq=50, node_rank=1)` worked.
- `get_state()` returned arm/head/lift/chassis state.
- `check_cams()` returned `(True, True, True)`.
- `get_cams([0, 1, 2])` returned three `480x640x3 uint8` frames.

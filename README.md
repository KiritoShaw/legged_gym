# Notes by Shaw_ZYX

建议先阅读 [minimal-isaac-gym](https://github.com/KiritoShaw/minimal-isaac-gym) 中的 `README.md` 文件，以及大致浏览一遍它的源码

## 项目结构概览

一级目录中主要有三个文件夹：legged_gym*，licenses 以及 resources。其中最重要的自然是 legged_gym，具体放在后面再介绍。而 resources 文件夹中
存放了各种资源，例如执行器网络，各种机器人的 URDF 文件以及 meshes 文件。

我们通过目录树介绍 legged_gym 的文件结构

* legged_gym
  * envs 
  * scripts
    * `play.py` 创建少量机器人的仿真环境，用于测试训练结果
    * `train.py` 训练
  * tests
    * `test_env.py`
  * utils
    * `helpers.py` 杂七杂八的东西
    * `logger.py` 在运行 `play.py` 时用于返回各种指标以及绘制图片。可以修改该文件进行自定义。
    * `math.py`
    * `task_registry.py` 用于注册任务以及通过命令行参数返回 {环境、环境配置、训练算法以及算法配置}
    * `terrain.py` 用于生成各种地形
  * `__init__.py` 保存了根目录路径




## train.py

从 utils 中导入 `get_args` 和 `task_registry`，前者用于解析命令行传入的参数，后者用于对注册的 RL 任务进行一些操作。
此时 task_registry 是 TaskRegistry 类的实例化对象。

```python
from legged_gym.utils import get_args, task_registry
```

`main` 函数中先将参数进行解析，然后将 `args` 作为参数输入函数 `train`。函数 `train` 负责三个功能：
1. 根据 `args` 中指定的任务名创建对应的 RL 任务环境 VecEnv `env` 并返回相应的配置 `env_cfg`
2. 根据环境 `env` 以及 `args` 创建训练算法并返回训练配置 `train_cfg`

```python
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
```

## Task_registry

Task_registry 类主要用在两个地方：

一是 `legged_gym/envs/__init__.py` 中将 RL 环境以及相应的环境配置以及训练配置注册成一个 RL 任务，

```python
from legged_gym.utils.task_registry import task_registry
from .anymal_c.anymal import Anymal  # RL 环境
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO  # RL 环境配置以及算法配置

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )  # 注册成任务
```

二是根据 `args` 创建对应的环境，环境配置，训练算法以及训练配置。如上一节所示。

这里创建的

* 环境可以参考 `legged_gym/envs/base/base_task.py` 中的 `BaseTask` 或者 `legged_gym/envs/base/legged_robot.py` 中的 
`LeggedRobot`（该类继承了基类 `BaseTask`） 
* 环境配置可以参考 `legged_gym/envs/base/legged_robot_config.py` 中的 `LeggedRobotCfg`（该类继承了基类 `BaseConfig`） 
* 训练算法 `***_runner` 是 `rsl_rl` 库中 `OnPolicyRunner` 的实例化
* 训练配置可以参考 `legged_gym/envs/base/legged_robot_config.py` 中的 `LeggedRobotPPOCfg`（该类同样继承了基类 `BaseConfig`）





---

Copied from `legged_gym/README.md`

# Isaac Gym Environments for Legged Robots #
This repository provides the environment used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym.
It includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.  
**Maintainer**: Nikita Rudin  
**Affiliation**: Robotic Systems Lab, ETH Zurich  
**Contact**: rudinn@ethz.ch  

### Useful Links ###
Project website: https://leggedrobotics.github.io/legged_gym/
Paper: https://arxiv.org/abs/2109.11978

### Installation ###
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
2. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone https://github.com/leggedrobotics/rsl_rl
   -  `cd rsl_rl && pip install -e .` 
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`

### CODE STRUCTURE ###
1. Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`). The config file contains two classes: one conatianing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).  
2. Both env and config classes use inheritance.  
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward.  
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `envs/__init__.py`, but can also be done from outside of this repository.  

### Usage ###
1. Train:  
  ```python issacgym_anymal/scripts/train.py --task=anymal_c_flat```
    -  To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    -  To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
    - The trained policy is saved in `issacgym_anymal/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    -  The following command line arguments override the values set in the config files:
     - --task TASK: Task name.
     - --resume:   Resume training from a checkpoint
     - --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
     - --run_name RUN_NAME:  Name of the run.
     - --load_run LOAD_RUN:   Name of the run to load when resume=True. If -1: will load the last run.
     - --checkpoint CHECKPOINT:  Saved model checkpoint number. If -1: will load the last checkpoint.
     - --num_envs NUM_ENVS:  Number of environments to create.
     - --seed SEED:  Random seed.
     - --max_iterations MAX_ITERATIONS:  Maximum number of training iterations.
2. Play a trained policy:  
```python issacgym_anymal/scripts/play.py --task=anymal_c_flat```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.

### Adding a new environment ###
The base environment `legged_robot` implements a rough terrain locomotion task. The corresponding cfg does not specify a robot asset (URDF/ MJCF) and no reward scales. 

1. Add a new folder to `envs/` with `'<your_env>_config.py`, which inherit from an existing environment cfgs  
2. If adding a new robot:
    - Add the corresponding assets to `resourses/`.
    - In `cfg` set the asset path, define body names, default_joint_positions and PD gains. Specify the desired `train_cfg` and the name of the environment (python class).
    - In `train_cfg` set `experiment_name` and `run_name`
3. (If needed) implement your environment in <your_env>.py, inherit from an existing environment, overwrite the desired functions and/or add your reward functions.
4. Register your env in `isaacgym_anymal/envs/__init__.py`.
5. Modify/Tune other parameters in your `cfg`, `cfg_train` as needed. To remove a reward set its scale to zero. Do not modify parameters of other envs!


### Troubleshooting ###
1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, do: `sudo apt install libpython3.8`

### Known Issues ###
1. The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors, but the force are propagated through the sensors of consecutive bodies resulting in an undesireable behaviour. However, for a legged robot it is possible to add sensors to the feet/end effector only and get the expected results. When using the force sensors make sure to exclude gravity from trhe reported forces with `sensor_options.enable_forward_dynamics_forces`. Example:
```
    sensor_pose = gymapi.Transform()
    for name in feet_names:
        sensor_options = gymapi.ForceSensorProperties()
        sensor_options.enable_forward_dynamics_forces = False # for example gravity
        sensor_options.enable_constraint_solver_forces = True # for example contacts
        sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
        index = self.gym.find_asset_rigid_body_index(robot_asset, name)
        self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)
    (...)

    sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
    self.gym.refresh_force_sensor_tensor(self.sim)
    force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
    self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
    (...)

    self.gym.refresh_force_sensor_tensor(self.sim)
    contact = self.sensor_forces[:, :, 2] > 1.
```

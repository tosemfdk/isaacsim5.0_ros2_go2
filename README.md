# isaacsim5.0_ros2_go2
The goal of this repository is to achieve full control of the Unitree Go2 quadruped robot within the Isaac Sim 5.0 simulation environment, utilizing ROS 2 Humble as the communication and control framework.
---

# How to install
## 1. Install IsaacSim 5.0 & isaac Lab 2.2.0
### isaacsim
- [isaacsim 5.0 installation](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/index.html)
> you need to install isaacsim in home directory (~/) and setup ros2_ws(humble) following instruction above


- [isaaclab 2.2.0 installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/source_installation.html)
copy and paste 
```bash
# 1. clone isaaclab repo
cd ~
git clone git@github.com:isaac-sim/IsaacLab.git

# 2. checkout to 2.2.0
cd IsaacLab
git checkout release/2.2.0

# 3. make conda env
./isaaclab.sh --conda isaaclab
conda activate isaaclab

# 4. source isaacsim
source ~/isaacsim/setup_conda_env.sh

# 4. install isaaclab
./isaaclab.sh --install
```




## 2. source isaacsim & isaacsim ros2_ws
### notice
>you shouldn't source your system ros2 ws because it is ros packages for 3.10 (ubuntu 22.04) or 3.12 (ubuntu24.04) ex) source /opt/ros/humble/setup.bash



```bash
# if you want to start isaaclab from your conda env, you need to source your isaacsim & ros2 ws

source ~/isaacsim/setup_conda_env.sh
# we need to source ros2 packages that is built with python 3.11 for isaacsim
source ~/IsaacSim-ros_workspaces/build_ws/humble/isaac_sim_ros_ws/install/local_setup.bash
source ~/IsaacSim-ros_workspaces/build_ws/humble/humble_ws/install/local_setup.bash

```

---
# How to start
```bash
# activate conda & source your isaacsim & isaacsim ros2 packages first

cd path/to/isaacsim5.0_ros2_go2
python run_simul.py
```


---
# Multi Robot Teleoperation
[스크린캐스트 01-06-2026 06:19:03 PM.webm](https://github.com/user-attachments/assets/f22e39df-9306-44a1-b699-f9839c981dbf)
[스크린캐스트 01-07-2026 12:06:04 PM.webm](https://github.com/user-attachments/assets/67bed969-d5be-4c21-b6e6-3f0aea80a846)

---
# Importing Go2 into a Digital Twin environment

https://github.com/user-attachments/assets/5931b57b-1448-4586-883c-942dbf16f299


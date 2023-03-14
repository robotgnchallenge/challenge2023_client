# challenge2023_client

## Introduction
In 2023, we host the Mobile Robot Grasping and Navigation Challenge that evaluates the robot's ability of scene understanding, navigation and grasping in the household scenario. The robot needs to explore a random environment, find a specific object and provide a stable grasp. CCF TCIR as the organizing committee launched the challenge. In this challenge, the organizer provides a downloadable simulator and an online evaluation system. Participants can develop their own scene understanding, navigation and grasping algorithms to complete the tasks. For more details, see https://robomani-challenge.bytedance.com/.

## Requirements
### pybind11
```shell
  cd ~/tools
  git clone https://github.com/pybind/pybind11.git
  cd pybind11/
  mkdir build
  cd build
  cmake ..
  cmake --build . --config Release  
  make 
```

### CUDA & cuDNN
* CUDA 11.1
* cuDNN 8.7.0



##### Troubleshooting
```shell
# ERROR1: /usr/bin/ld: cannot find -l*
sudo ln -s ${pybind11_path}/pybind11/build/tests/pybind11_cross_module_tests.cpython-${python_version}-x86_64-linux-gnu.so /usr/local/lib/libpybind11_cross_module_tests.cpython-${python_version}-x86_64-linux-gnu.so
sudo ln -s ${pybind11_path}/pybind11/build/tests/pybind11_tests.cpython-${python_version}-x86_64-linux-gnu.so /usr/local/lib/libpybind11_tests.cpython-${python_version}-x86_64-linux-gnu.so

# ERROR2: usr/lib/x86_64-linux-gnu/libapr-1.so.0：对‘uuid_generate@UUID_1.0’未定义的引用
sudo rm ${path}/anaconda3/envs/${your_env}/lib/libuuid.so.1
sudo ln -s /lib/x86_64-linux-gnu/libuuid.so.1 ${path}/anaconda3/envs/${your_env}/lib/libuuid.so.1
```

### Build Environment
```shell
cd ~
git clone https://github.com/robotgnchallenge/challenge2023_client.git
mv challenge2023_client client_ws
conda activate robotmani
cd ~/client_ws/src/task_client/scripts/graspnetAPI
pip install .

cd ~/client_ws
vim ~/client_ws/src/task_client/CMakeLists.txt
# Please Comment out lines from 117 to 131
catkin build
vim ~/client_ws/src/task_client/CMakeLists.txt
# Uncomment lines from 117 to 131
catkin build
```

##### Troubleshooting
```shell
# If error occurs, please check ~/client_ws/src/task_client/CMakeLists.txt first, make sure enviroment path are correct in line 117 to 131

# ERROR1: Python.h: No such file or directory
vim ~/client_ws/src/task_client/CMakeLists.txt
# Modify line 117, 118, 120, 129 to match your anaconda path

# ERROR2: pybind11/* : No such file or directory
vim ~/client_ws/src/task_client/CMakeLists.txt
# Modfy line 119, 121, 122 to match your pybind11 path

# ERROR2: pcl/* : No such file or directory
sudo apt install libpcl-dev
```

### Task Server
Please follow https://robomani-challenge.bytedance.com/tutorial to install task server.

### Demo
##### Run Server
```shell
# In this step, we assume the task server is already build
# Terminal 1
cd ~/server_ws/src/webots_ros
./server_run.sh
```
##### Run Client
```shell
# Terminal 2
cd ~/client_ws
source devel/setup.bash
cd src/task_client/demo/src
conda activate robotmani
python object_search_planner_socket.py

# Terminal 3
cd ~/client_ws
source devel/setup.bash
conda activate robotmani
rosrun task_client construct_semmap_with_socket

# Terminal 4
cd ~/client_ws
source devel/setup.bash
cd src/task_client/demo/src
conda activate robotmani
python task_client_demo.py
```

##### Troubleshooting
```shell
# ERROR1: version `GLIBCXX_3.4.29' not found
Please refer to https://blog.csdn.net/weixin_39379635/article/details/129159713

# ERROR2: Grasp Detection failed
Please configure CUDA, cuDNN, and tensorflow-gpu environments that match your NVIDIA GPU.

# ERROR3: ROS_MASTER_URI port [10241] does not match this roscore [11311]
# [IMPORTANT] Only for debugging and development, your submission docker should run in an isolated ROS master
export ROS_MASTER_URI=http://127.0.0.1:10241

```







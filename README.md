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

### Task Server
Please follow https://robomani-challenge.bytedance.com/tutorial to install task server.

##### Troubleshooting
```shell
#ERROR1: /usr/bin/ld: cannot find -l*
sudo ln -s ${path}/pybind11/build/tests/pybind11_cross_module_tests.cpython-37m-x86_64-linux-gnu.so /usr/local/lib/libpybind11_cross_module_tests.cpython-37m-x86_64-linux-gnu.so
sudo ln -s ${path}/pybind11/build/tests/pybind11_tests.cpython-37m-x86_64-linux-gnu.so /usr/local/lib/libpybind11_tests.cpython-37m-x86_64-linux-gnu.so

# ERROR2: usr/lib/x86_64-linux-gnu/libapr-1.so.0：对‘uuid_generate@UUID_1.0’未定义的引用
sudo rm ${path}/anaconda3/envs/<yourenv>/lib/libuuid.so.1
sudo ln -s /lib/x86_64-linux-gnu/libuuid.so.1 ${path}/anaconda3/envs/${your_env}/lib/libuuid.so.1
```

### Build Enviroment
```shell
cd ~
git clone https://github.com/robotgnchallenge/challenge2023_client.git
mv challenge2023_client client_ws
conda activate robotmani
cd ~/client_ws/src/task_client/scripts/graspnetAPI
pip install .

cd ~/client_ws
catkin build
```

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







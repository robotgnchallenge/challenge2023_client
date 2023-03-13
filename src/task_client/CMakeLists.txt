cmake_minimum_required(VERSION 2.8.3)
project(task_client)
set(CMAKE_CXX_STANDARD 17)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(catkin REQUIRED COMPONENTS 
                roscpp 
                rospy 
                std_msgs 
                sensor_msgs 
                message_generation 
                tf 

                actionlib 
                actionlib_msgs 
                
                tf_conversions
                pcl_conversions
                cv_bridge
                message_filters
                message_filters)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
  add_message_files(
    FILES
    ObjectGoal.msg
    feature.msg
    InstanceInfo.msg
  )

## Generate services in the 'srv' folder
  add_service_files(
    FILES
    attach_obj.srv
    search_and_pick_obj.srv
    get_bool.srv
    get_float_array.srv
    get_float.srv
    get_int.srv
    get_string.srv
    get_uint64.srv
    get_urdf.srv
    set_bool.srv
    set_float.srv
    set_float_array.srv
    set_int.srv
    set_string.srv
    get_grasp_pose.srv
  )

  add_action_files(
     FILES
     AskForSearchObj.action
     LookAt.action
  )

## Generate added messages and services with any dependencies listed here
  generate_messages(
    DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
    actionlib_msgs
  )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs message_runtime tf
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY plugins
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

#pybind compilation
link_directories($ENV{HOME}/anaconda3/envs/robotmani/lib)
INCLUDE_DIRECTORIES($ENV{HOME}/anaconda3/envs/robotmani/include/python3.8)
INCLUDE_DIRECTORIES($ENV{HOME}/tools/pybind11/include)
link_libraries($ENV{HOME}/anaconda3/envs/robotmani/lib/libpython3.8.so)
link_libraries($ENV{HOME}/tools/pybind11/build/tests/pybind11_cross_module_tests.cpython-38-x86_64-linux-gnu.so)
link_libraries($ENV{HOME}/tools/pybind11/build/tests/pybind11_tests.cpython-38-x86_64-linux-gnu.so)
add_executable(construct_semmap_with_socket demo/src/construct_semmap_with_socket.cpp)
target_include_directories(construct_semmap_with_socket PUBLIC demo/include)
target_link_libraries(construct_semmap_with_socket
         ${catkin_LIBRARIES}
         ${OpenCV_LIBRARIES}
         ${PCL_LIBRARIES}
        "$ENV{HOME}/anaconda3/envs/robotmani/lib/libtbb.so.2"
        "/usr/lib/x86_64-linux-gnu/libtbb.so.2"
         )

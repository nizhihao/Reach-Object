# Reach Object

## Installation

This implementation requires the following dependencies (tested on Ubuntu 16.04 LTS):

- Python 2.7. The default python version in the ubuntu 16.04.

- [ROS Kinetic](http://wiki.ros.org/Installation/Ubuntu). You can quickly install the ROS and Gazebo by following the wiki installation web(if you missing dependency package, you can install these package by runing the following):

  ```shell
  sudo apt-get install ros-kinetic-(Name)
  sudo apt-get install ros-kinetic-(the part of Name)*   # * is the grammar of regular expression
  ```

- [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/scipylib/index.html), [OpenCV-Python](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html), [Matplotlib](https://matplotlib.org/). You can quickly install/update these dependencies by running the following:

  ```shell
  pip install numpy scipy opencv-python matplotlib
  ```

- Torch == 1.0.0 and Torchvision == 0.2.1

  ```shell
  pip install torch==1.0.0 torchvision==0.2.1
  ```

- CUDA and cudnn. You need to install the GPU driver、the cuda and the cudnn, this code has been tested with CUDA 9.0 and cudnn 7.1.4 on two 1080Ti GPU(11GB).

## Reach Object Code

1. download this repository and compile the ROS workspace.

   ```shell
   git clone https://github.com/nizhihao/Reach-Object.git
   mv /home/user/Reach-Object/myur_ws /home/user
   cd /home/user/myur_ws
   catkin_make
   echo "source /home/user/myur_ws/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

   The Introduction of the ROS package.

   ```shell
   gazebo-pkgs       # Gazebo grasp plugin
   reach_objcet    # the RL package(all algorithm code in this package)
   universal_robot-kinetic-devel  	# UR robotic arm package
   ur_modern_driver 		     # UR robotic arm driver in the real world 
   ur_robotiq 			  # cube, Mesh
   ```
   
2. If you want to train or test in Gazebo,  You can run the following command line.

   ```shell
roslaunch ur_gazebo ur5.launch   # run the Gazebo
   roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch sim:=true   # run moveit
   roslaunch ur5_moveit_config moveit_rviz.launch config:=true  # run rviz
   rosrun reach_object main.py   # run the agent
   ```
   
   The Introduction of the reach_object(RL) package.
   
   ```shell
   main.py   	   # main func
   TD3.py   	 # agent 
   env.py   # environment
   plot.py		   # plot
   ```
   
   


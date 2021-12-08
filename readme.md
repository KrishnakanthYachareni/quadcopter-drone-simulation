To access project and execution logon to https://www.theconstructsim.com/the-ros-development-studio-by-the-construct/

ROS Development Studio
-----------------------
Username: krishnakanthy
Password: <common> 

This is a simple drone simulation application developed using gazebo and ros packages. This drone can be controlled using computer keyboard.

Demo: https://www.youtube.com/watch?v=1vxK8iYLOPc

TODO: Still in progress.

#### Envornment Setup
1. Linux:    Ubuntu 16.04.6 LTS (Codename: xenial)
2. ROS-kinetic distribution

**Step-1:**
````bash
$ cd <catkin_ws>/src
$ git clone https://github.com/KrishnakanthYachareni/quadcopter-drone-simulation.git
$ cd <catkin_ws>
$ catkin_make
````
**Step-2:**
```python
roslaunch drone_ring_tests main.launch
```

**To Move Drone**
```python
rostopic pub /drone/takeoff std_msgs/Empty "{}"
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```
**Control Drone with Keys**
````
Reading from the keyboard  and Publishing to Twist!
---------------------------
Moving around:
   u    i    o
   j    k    l
   m    ,    .

For Holonomic mode (strafing), hold down the shift key:
---------------------------
   U    I    O
   J    K    L
   M    <    >

t : up (+z)
b : down (-z)

anything else : stop

q/z : increase/decrease max speeds by 10%
w/x : increase/decrease only linear speed by 10
````

Launch The Different trackers for each goal you want to track, you can put them all in the same launch or in separate launches:


```python
roslaunch models_world_tracker model_tracker.launch
roslaunch models_world_tracker model_tracker_2.launch
```

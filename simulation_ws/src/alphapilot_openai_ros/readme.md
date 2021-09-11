# AlphaPilot OpenAI ROS Prototype - ROS Kinetic, Gazebo 7

This repository contains code for an AlphaPilot OpenAI ROS prototype implemented using ROS Kinetic and Gazebo 7.

## Setup

### Running in a local environment.

#### Clone required sources into a catkin workspace

Create a `ros-kinetic-alphapilot` catkin workspace:
```bash
export ROS_DISTRO=kinetic
source /opt/ros/$ROS_DISTRO/setup.bash
export WORKSPACE=~/mount/project/ros-kinetic-alphapilot/catkin_ws
mkdir -p $WORKSPACE/src
cd $WORKSPACE
catkin init
```

Clone sources into the `catkin_ws/src` folder:
```bash
cd $WORKSPACE/src
git clone https://github.com/edowson/alphapilot_openai_ros
git clone https://github.com/edowson/openai_ros
git clone https://github.com/edowson/sjtu_drone
```

Identify ROS package dependencies:
```bash
cd $WORKSPACE
sudo apt update
rosdep install --from-paths src --ignore-src --rosdistro=$ROS_DISTRO -r -y
```

---

### Running locally using docker containers

You will have to first clone the top-level `alphapilot-ros-kinetic-gazebo7` project and build the docker image. The docker image will automatically pull and build the sources contained in this repository.

**Note:** The `alphapilot-ros-kinetic-gazebo7` top-level github repo is currently private.


#### Clone project sources

Create a project folder:
```bash
mkdir -p cs234/2019-winter/project
cd cs234/2019-winter/project/
```

Clone the top-level`alphapilot-ros-kinetic-gazebo7` project:
```bash
git clone --recurse-submodules -j4 https://github.com/edowson/alphapilot-ros-kinetic-gazebo7 alphapilot-ros-kinetic-gazebo7
```

#### Build docker image

Build the `ros-alphapilot` docker image:
```bash
cd alphapilot-ros-kinetic-gazebo7/docker/ros/kinetic/ubuntu/xenial/ros-alphapilot
./build-ros-alphapilot-gazebo.sh
```

This will create a docker image called `edowson/ros-alphapilot/gazebo:kinetic-xenial`

You should be able to see a prompt in the terminal console with the user `developer`:
```bash
developer:~$ pwd
/home/developer
```

#### Run docker container

Run the `ros-alphapilot` docker image:
```
./run-alphapilot-gazebo.sh
```

This script also configures port forwarding from your local interface to allow
access to tensorboard running within the docker container.

```bash
-p 127.0.0.1:6006:6006
```

This will create and launch a docker container named `ros-alphapilot-gazebo-kinetic-xenial`.

#### Shutdown docker container

Shutdown the `ros-alphapilot-gazebo-kinetic-xenial` docker container:
```bash
./shutdown-alphapilot-gazebo.sh
```

---

## Build the project workspace

Build the sources in the `ros-kinetic-alphapilot` workspace:
```bash
cd $WORKSPACE
catkin_make -j4
```

---

## Launch gazebo simulations

### 01. Parrot AR Drone

#### Launch the `ardrone_race_track` simulation


Shell 02:
```bash
source $WORKSPACE/devel/setup.bash

# launch simulation using race track 01
roslaunch ardrone_race_track start_simulation.launch race_track:=01

# launch simulation using race track 02
roslaunch ardrone_race_track start_simulation.launch race_track:=02
```

#### Launch the `ardrone_race_track` agent training

Shell 03:
```bash
source $WORKSPACE/devel/setup.bash

# train agent using DQN model
roslaunch ardrone_race_track start_training.launch model:='ddqn'

# train agent using SNAIL model
roslaunch ardrone_race_track start_training.launch model:='snail'
```

To run tensorboard:
```bash
cd $WORKSPACE/src/alphapilot_openai_ros/ardrone_race_track
tensorboard --logdir ./output --host 0.0.0.0 --port 6006
```

You can access tensorboard from the host by going to:
```
http://127.0.0.1:6006
```


#### Manually launch the ardrone

Send a takeoff message:

Shell 04:
```bash
rostopic pub /drone/takeoff std_msgs/Empty "{}"
```

Press `ctrl+c` to quit the command publisher.


#### Move the robot using the keyboard

You can move the fetch robot using the `teleop_twist_keyboard.py` script:

Shell 04:
```bash
source $WORKSPACE/devel/setup.bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

This will display the following output with instructions for controlling the robot
using the keyboard.

```bash
Use
i: for moving forward
,: for moving backward
u: for turning left
o: for turning right
```

```bash
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
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only angular speed by 10%

CTRL-C to quit
```

You can type `kill-ros` to kill all ros services and processes.
This has been conveniently defined as an alias in the `~/.bashrc` file.
```bash
alias kill-ros="killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient"
```

#### Display drone camera images

Run the following commands to view the images from the front and downward facing
cameras.

Shell 05:
```bash
rosrun image_view image_view image:=/drone/front_camera/image_raw
rosrun image_view image_view image:=/drone/down_camera/image_raw
```

---

## Model Preparation

### Race gates

#### Include a race gate in a world model

Include a race gate model in a world file.
```xml
    <include>
      <uri>model://race_gate_square_h1</uri>
      <name>race_gate_square_h1</name>
      <pose>6.0 0.0 0.0 0 0 1.570796</pose>
      <static>true</static>
    </include>
```
Note:
- Make the race gate static by setting the `static` tag to true.
  This will prevent the gate from wobbling if a drone collides with it.

#### Gazebo model preparation for collision detection.

In order to make collision detection work, you will have to ensure that:
- your models have a unique name for the link and collision tags.
- the `collision` tag in the contact sensor matches the name of the collision tag of the link

##### Adding a contact sensor plugin to a drone

In this example, we modify the modify the `race_track_01.world` file to add a gazebo ros bumper sensor to the `sjtu_drone`.
```xml
   <!-- sjtu drone -->
   <model name='sjtu_drone'>
      <pose frame=''>0 0 1 0 -0 0</pose>
      <link name='base_link'>
        <inertial>
          <mass>1.477</mass>
          <pose frame=''>0 0 0 0 -0 0</pose>
          <inertia>
            <ixx>0.1152</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.1152</iyy>
            <iyz>0</iyz>
            <izz>0.218</izz>
          </inertia>
        </inertial>
        <visual name='visual'>
          <geometry>
            <mesh>
              <uri>file://models/sjtu_drone/quadrotor_4.dae</uri>
            </mesh>
          </geometry>
        </visual>
        <collision name='sjtu_drone_collision'>
          <geometry>
            <mesh>
              <uri>file://models/sjtu_drone/quadrotor_4.dae</uri>
            </mesh>
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact>
              <ode/>
            </contact>
            <bounce/>
            <friction>
              <torsional>
                <ode/>
              </torsional>
              <ode/>
            </friction>
          </surface>
        </collision>
        <gravity>1</gravity>

        <!-- sjtu drone: contact sensor -->
        <sensor name="sjtu_drone_contact_sensor" type="contact">
          <update_rate>1000.0</update_rate>
          <always_on>true</always_on>
          <contact>
            <collision>sjtu_drone_collision</collision>
          </contact>
          <plugin name="sjtu_drone_gazebo_ros_bumper_sensor" filename="libgazebo_ros_bumper.so">
            <update_rate>1000.0</update_rate>
            <always_on>true</always_on>
            <bumperTopicName>/drone/contacts</bumperTopicName>
            <frameName>base_link</frameName>
          </plugin>
        </sensor>

      </link>
    </model>
```

Launch the gazebo world and check for the new topic:
```bash
$ rostopic list

/drone/contacts
```

Check the rostopic:
```bash
rostopic echo /drone/contacts | grep collision
```

If the drone hits a race gate. you should see the following message:
```bash
  info: "Debug:  i:(0/1)     my geom:race_gate_square_h1::race_gate_square_h1::race_gate_square_h1_collision\
  \   other geom:sjtu_drone::base_link::sjtu_drone_collision         time:3249.679000000\n"
  collision1_name: "race_gate_square_h1::race_gate_square_h1::race_gate_square_h1_collision"
  collision2_name: "sjtu_drone::base_link::sjtu_drone_collision"
```

##### Adding a contact sensor plugin to a race gate

In this example, we modify the race gate model and add a gazebo ros bumper sensor.

```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <!-- race gate square h1 -->
  <model name="Race Gate Square H1">
    <link name="race_gate_square_h1">
      <collision name="race_gate_square_h1_collision">
        <geometry>
          <mesh>
            <uri>model://race_gate_square_h1/meshes/05.dae</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name="race_gate_square_h1">
        <geometry>
          <mesh>
            <uri>model://race_gate_square_h1/meshes/05.dae</uri>
          </mesh>
        </geometry>
      </visual>
      <!-- race gate square h1: contact sensor -->
      <sensor name="race_gate_square_h1_contact_sensor" type="contact">
        <update_rate>1000.0</update_rate>
        <always_on>true</always_on>
        <contact>
          <collision>race_gate_square_h1_collision</collision>
        </contact>
        <plugin name="race_gate_square_h1_gazebo_ros_bumper_sensor" filename="libgazebo_ros_bumper.so">
          <update_rate>1000.0</update_rate>
          <always_on>true</always_on>
          <bumperTopicName>/race_gate_square_h1_contacts</bumperTopicName>
          <frameName>race_gate_square_h1</frameName>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

Note: Add collision sensors to individual race gates is not necessary from a simulation task point of view.
Adding a collision sensor to the drone is sufficient. However, it may be required if a drone race sponsor
has no access to drone telemetry from on-board sensors to determine which drones collided with specific race gates.

Launch the gazebo world and check for the new topic:
```bash
$ rostopic list

/race_gate_square_h1_contacts
```

Check the rostopic for collision detections:
```bash
rostopic echo /race_gate_square_h1_contacts | grep collision
```

---

## Repositories

01. [sjtu_drone - github/edowson](https://github.com/edowson/sjtu_drone)

Related repositories:
- [parrot_ardrone - bitbucket/theconstructcore](https://bitbucket.org/theconstructcore/parrot_ardrone)
- [sjtu_drone - bitbucket/dannis](https://bitbucket.org/dannis/sjtu_drone)
- [tum_simulator - github/tum-vision](https://github.com/tum-vision/tum_simulator)

---

## Technotes

01. [tum_simulator](http://wiki.ros.org/tum_simulator)

02. [Collide bitmask - Gazebo Tutorials](http://gazebosim.org/tutorials?tut=collide_bitmask&cat=physics)

---

## Tutorials

01. [Installing and Starting Gazebo](http://wiki.ros.org/simulator_gazebo/Tutorials/StartingGazebo)

---

## Related Topics

01. [View Tensorboard on Docker on Google Cloud](https://stackoverflow.com/questions/33836728/view-tensorboard-on-docker-on-google-cloud)

02. [Docker compose port mapping](https://stackoverflow.com/questions/35429837/docker-compose-port-mapping)

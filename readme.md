Launch Sim


```python
roslaunch drone_ring_test main.launch
```

Move Drone


```python
rostopic pub /drone/takeoff std_msgs/Empty "{}"
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

Launch The Different trackers for each goal you want to track, you can put them all in the same launch or in separate launches:


```python
roslaunch models_world_tracker model_tracker.launch
roslaunch models_world_tracker model_tracker_2.launch
```

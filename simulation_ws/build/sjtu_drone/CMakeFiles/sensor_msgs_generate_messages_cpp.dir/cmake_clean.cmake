file(REMOVE_RECURSE
  "plugin_ros_sonar_automoc.cpp"
  "spawn_drone_automoc.cpp"
  "drone_keyboard_automoc.cpp"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/sensor_msgs_generate_messages_cpp.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

file(REMOVE_RECURSE
  "plugin_ros_sonar_automoc.cpp"
  "spawn_drone_automoc.cpp"
  "drone_keyboard_automoc.cpp"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/geometry_msgs_generate_messages_py.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()

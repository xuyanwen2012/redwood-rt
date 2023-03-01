
foreach(BACKEND_NAME IN LISTS BACKENDS)
  get_filename_component(BIN_BASE_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
  set(BIN_NAME ${BIN_BASE_NAME}_${BACKEND_NAME})

  add_executable(${BIN_NAME}
    Main.cpp
  )

  target_link_libraries(${BIN_NAME}
    redwood_${BACKEND_NAME}
  )

  add_custom_command(TARGET ${BIN_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
    ${CMAKE_CURRENT_BINARY_DIR}/${BIN_NAME}
    ${PROJECT_BINARY_DIR}/${BIN_NAME}
  )
endforeach()

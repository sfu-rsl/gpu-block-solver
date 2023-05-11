add_library(gpu_solver ${G2O_LIB_TYPE}
    block_solver2.h
    block_solver2.hpp
    BAStats.hpp
    BAStats.cpp
)

set_target_properties(gpu_solver PROPERTIES OUTPUT_NAME ${LIB_PREFIX}gpu_solver)
set_target_properties(gpu_solver PROPERTIES
  VERSION ${G2O_LIB_VERSION}
  SOVERSION ${G2O_LIB_SOVERSION})
if (APPLE)
  set_target_properties(gpu_solver PROPERTIES INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()
target_compile_features(gpu_solver PUBLIC cxx_std_17)
target_link_libraries(gpu_solver core)

target_include_directories(gpu_solver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include/g2o/gpu>
)

install(TARGETS gpu_solver
  EXPORT ${G2O_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
  INCLUDES DESTINATION ${INCLUDES_DESTINATION}
)

file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
install(FILES ${headers} DESTINATION ${INCLUDES_INSTALL_DIR}/gpu)


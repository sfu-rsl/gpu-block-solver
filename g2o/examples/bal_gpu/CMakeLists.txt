add_executable(bal_gpu bal_gpu.cpp)
if(CHOLMOD_FOUND)
  target_link_libraries(bal_gpu solver_cholmod gpu_solver compute)
else()
  target_link_libraries(bal_gpu solver_eigen gpu_solver compute)
endif()


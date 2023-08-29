add_executable(bal_gpu bal_gpu.cpp)
if(CHOLMOD_FOUND)
  target_link_libraries(bal_gpu solver_cholmod gpu_solver compute)
else()
  target_link_libraries(bal_gpu solver_eigen gpu_solver compute)
endif()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  message("bal_gpu: CUDA IS AVAILABLE")

  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(bal_gpu CUDA::cusparse CUDA::cusolver CUDA::cudart)
  add_compile_definitions(CUDA_ENABLED)
else()
  message("bal_gpu: CUDA IS NOT AVAILABLE")
endif()
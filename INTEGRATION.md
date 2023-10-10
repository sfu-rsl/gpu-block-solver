## Using the GPU block solver with ORB-SLAM3

Note: See [here](https://github.com/sfu-rsl/compute-engine#building) for more information on required dependencies to build the `compute-engine` submodule. 

1. Clone the repositories and copy the block solver implementation:
    ```bash
    git clone git@github.com:sfu-rsl/gpu-block-solver.git
    git clone git@github.com:UZ-SLAMLab/ORB_SLAM3.git
    cd ORB_SLAM3/Thirdparty/g2o/
    git submodule add -f git@github.com:sfu-rsl/compute-engine.git
    git submodule update --init --recursive
    cp -r ../../../gpu-block-solver/g2o/gpu/ g2o/
    rm g2o/gpu/CMakeLists.txt
    ```


2. At the top of block_solver2.h, replace `#include "g2o/config.h"` with `#include "../../config.h"`.



3. The dependencies need at least C++14, so you need to edit all the CMakeLists.txt files:
    - `ORB_SLAM3/CMakeLists.txt`
    - `ORB_SLAM3/Thirdparty/Sophus/CMakeLists.txt`
    - `ORB_SLAM3/Examples_old/ROS/ORB_SLAM3/CMakeLists.txt`

    In those files you need to replace `c++11` with `c++14` and `CXX11` with `CXX14`. Also change `set(CMAKE_CXX_STANDARD 11)` to `set(CMAKE_CXX_STANDARD 14)`. The definition `DCOMPILEDWITHC11` can stay the same, otherwise you will need to change the use of that definition in the code as well.

4. In `ORB_SLAM3/Thirdparty/g2o/CMakeLists.txt` make the following changes:

    ```CMake
    ...
    # Set up the top-level include directories
    INCLUDE_DIRECTORIES(
    ${g2o_SOURCE_DIR}/core
    ${g2o_SOURCE_DIR}/types
    ${g2o_SOURCE_DIR}/stuff
    ${g2o_SOURCE_DIR}/gpu # new
    ${G2O_EIGEN3_INCLUDE})

    # new
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
    add_subdirectory(compute-engine)

    ...
    ```
    and later
    ```CMake
    ...
    # Include the subdirectories
    ADD_LIBRARY(g2o ${G2O_LIB_TYPE}
    #types
    g2o/types/types_sba.h
    ...
    g2o/stuff/property.cpp       
    g2o/stuff/property.h    
    # new
    g2o/gpu/BAStats.hpp
    g2o/gpu/BAStats.cpp
    g2o/gpu/block_solver2.h
    g2o/gpu/block_solver2.hpp
    )
    # new
    target_link_libraries(g2o compute)
    ```

5. In `ORB_SLAM3/CMakeLists.txt`, make the following changes.

    First, after line 29, add the following:
    ```CMake
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
    add_compile_definitions(KOMPUTE_VK_API_MAJOR_VERSION=1 KOMPUTE_VK_API_MINOR_VERSION=2)
    add_compile_definitions(KOMPUTE_DISABLE_VK_DEBUG_LAYERS)
    ```

    Then change `include_directories` for additional headers:
    ```CMake
    include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    # new
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/src/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/external/kompute/src/include
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/compute-engine/external/kompute/external/VulkanMemoryAllocator/include
    )
    ```

    Next, replace `add_subdirectory(Thirdparty/g2o)` with:

    ```CMake
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
    add_subdirectory(Thirdparty/g2o)

    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
    message("CUDA IS AVAILABLE")

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(${PROJECT_NAME} CUDA::cusparse CUDA::cusolver CUDA::cudart)
    add_compile_definitions(CUDA_ENABLED)
    else()
    message("CUDA IS NOT AVAILABLE")
    endif()
    ```

    Add `compute` and `fmt::fmt` to `target_link_libraries`:

    ```CMake
    target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
    -lboost_serialization
    -lcrypto
    # new
    compute
    fmt::fmt
    )
    ```

6. Add the following to `Optimizer.cc`:



    ```cpp

    ...

    #include "Thirdparty/g2o/g2o/gpu/block_solver2.h"

    ...

    namespace ORB_SLAM3
    {
    
    static compute::ComputeEngine* engine = nullptr;

    void initialize_compute_engine() {
        if (!engine) {
            engine = new compute::ComputeEngine();
        }

        // Temporary allocation to reduce initialization overhead on first use
        auto buf = engine->create_buffer<double>(nullptr, 1000000, compute::BufferType::DeviceCached);
        auto buf2 = engine->create_buffer<double>(nullptr, 1000000, compute::BufferType::Host);
        auto buf3 = engine->create_buffer<double>(nullptr, 1000000, compute::BufferType::Storage);

    }

    void destroy_compute_engine() {
        delete engine;
        engine = nullptr;
    }

    ...
    }
    ```

    and in `Optimizer.h` inside the ORB_SLAM3 namespace, add:
    ```cpp
    void initialize_compute_engine();
    void destroy_compute_engine();
    ```

7. Lastly, in `System.cc`, make the following changes.

    First, make sure to include this header:
    ```cpp
    #include "Optimizer.h"
    ```

    Then in the constructor around line 183:
    ```cpp
    if (mSensor==IMU_STEREO || mSensor==IMU_MONOCULAR || mSensor==IMU_RGBD)
        mpAtlas->SetInertialSensor();

    initialize_compute_engine(); // new
    ```
    Next, in the `Shutdown()` function, uncomment the following:
    ```cpp
    if(mpViewer)
    {
    mpViewer->RequestFinish();
    while(!mpViewer->isFinished())
        usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
    if(!mpLocalMapper->isFinished())
        cout << "mpLocalMapper is not finished" << endl;
    if(!mpLoopCloser->isFinished())
        cout << "mpLoopCloser is not finished" << endl;
    if(mpLoopCloser->isRunningGBA()){
        cout << "mpLoopCloser is running GBA" << endl;
        cout << "break anyway..." << endl;
        break;
    }
    usleep(5000);
    }
    ```

    Lastly, at the end of the `Shutdown()` function, add:
    ```cpp
    destroy_compute_engine();
    ```

## Using the block solver

You should now be able to replace BlockSolverX for BA problems, e.g.:

```cpp
// g2o::BlockSolverX::LinearSolverType * linearSolver;
// linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();

// g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

auto linearSolver = new compute::LDLTSolver<double>();
g2o::BlockSolver2X * solver_ptr = new g2o::BlockSolver2X(engine, linearSolver);
```

For Jetson devices, extra allocations can be avoided by setting:
```cpp
solver_ptr->setAllocType(compute::BufferType::Host);
```
Additionally `jetson_clocks` can be used to lock the frequencies.

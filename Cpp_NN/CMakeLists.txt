cmake_minimum_required(VERSION 3.10)

project(Cpp_NN)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_BUILD_TYPE Debug)   # This ensures the build is in debug mode
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")  # Adds debug symbols (-g)

include_directories(${PROJECT_SOURCE_DIR}/include)

## Enable AddressSanitizer if supported
#option(ENABLE_ASAN "Enable AddressSanitizer" ON)
#
#if(ENABLE_ASAN)
#    message(STATUS "AddressSanitizer enabled")
#
#    set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
#
#    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
#    add_link_options(-fsanitize=address)
#endif()

set(SOURCES 
    src/Matrix.cpp
    src/Layer.cpp
    src/Neural_Network.cpp
    src/Matrix_operations.cpp
    main_process.cpp
  )

add_executable(Cpp_NN ${SOURCES})

install(TARGETS Cpp_NN DESTINATION ${CMAKE_SOURCE_DIR})

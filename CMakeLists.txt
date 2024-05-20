cmake_minimum_required(VERSION 3.19)
project(CUDA_LEARNING LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

# ensure cuda is available
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 89)


# in debug mode, add debug symbols to device code
# this disables most optimizations and kills performance
add_compile_options( "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3;>")

#
# add_custom_target(generate_ptx ALL
#     DEPENDS kernel.ptx
# )

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

# Configure the source file path to be compiled
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# generate executable
add_executable(main main.cu ${SRC})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})


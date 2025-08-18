cmake_minimum_required(VERSION 3.18)

project(CudaPatterns LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

# --- Shared Libraries ---

# My own small library for graph construction, representation conversion and verification
add_library(graph_lib common/graph/graph.cpp common/graph/graph.h)
target_include_directories(graph_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

# stb library (https://github.com/nothings/stb)
add_library(stb_headers INTERFACE)
target_include_directories(stb_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common)

# --- Executables ---

add_executable(01-vector-add projects/01-vector-add/01-vector-add.cu)
add_executable(02-matrix-add projects/02-matrix-add/02-matrix-add.cu)
add_executable(03-image-blur projects/03-image-blur/03-image-blur.cu)
target_link_libraries(03-image-blur PRIVATE stb_headers)
add_executable(04-matrix-multiply-naive projects/04-matrix-multiply-naive/04-matrix-multiply-naive.cu)
add_executable(05-matrix-multiply-tiled projects/05-matrix-multiply-tiled/05-matrix-multiply-tiled.cu)
add_executable(06-matrix-multiply-tiled-coarsened projects/06-matrix-multiply-tiled-coarsened/06-matrix-multiply-tiled-coarsened.cu)
add_executable(07-convolution-naive projects/07-convolution-naive/07-convolution-naive.cu)
add_executable(08-convolution-tiled projects/08-convolution-tiled/08-convolution-tiled.cu)
add_executable(09-convolution-tiled-hybrid-cache projects/09-convolution-tiled-hybrid-cache/09-convolution-tiled-hybrid-cache.cu)
add_executable(10-stencil-sweep-naive projects/10-stencil-sweep-naive/10-stencil-sweep-naive.cu)
add_executable(11-stencil-sweep-tiled projects/11-stencil-sweep-tiled/11-stencil-sweep-tiled.cu)
add_executable(12-stencil-sweep-tiled-plane-sweep projects/12-stencil-sweep-tiled-plane-sweep/12-stencil-sweep-tiled-plane-sweep.cu)
add_executable(13-parallel-histogram-naive projects/13-parallel-histogram-naive/13-parallel-histogram-naive.cu)
add_executable(14-parallel-histogram-privatized projects/14-parallel-histogram-privatized/14-parallel-histogram-privatized.cu)
add_executable(15-parallel-histogram-shared projects/15-parallel-histogram-shared/15-parallel-histogram-shared.cu)
add_executable(16-parallel-histogram-shared-coarsened projects/16-parallel-histogram-shared-coarsened/16-parallel-histogram-shared-coarsened.cu)
add_executable(17-parallel-histogram-hybrid projects/17-parallel-histogram-hybrid/17-parallel-histogram-hybrid.cu)
add_executable(18-parallel-reduction projects/18-parallel-reduction/18-parallel-reduction.cu)
add_executable(19-prefix-sum-single-block projects/19-prefix-sum-single-block/19-prefix-sum-single-block.cu)
add_executable(20-prefix-sum-single-block-coarsened projects/20-prefix-sum-single-block-coarsened/20-prefix-sum-single-block-coarsened.cu)
add_executable(21-prefix-sum-general projects/21-prefix-sum-general/21-prefix-sum-general.cu)
add_executable(22-sparse-matrix-vector-coo projects/22-sparse-matrix-vector-coo/22-sparse-matrix-vector-coo.cu)
add_executable(23-sparse-matrix-vector-ell projects/23-sparse-matrix-vector-ell/23-sparse-matrix-vector-ell.cu)
add_executable(24-bfs-naive projects/24-bfs-naive/24-bfs-naive.cu)
target_link_libraries(24-bfs-naive PRIVATE graph_lib)
add_executable(25-bfs-frontier projects/25-bfs-frontier/25-bfs-frontier.cu)
target_link_libraries(25-bfs-frontier PRIVATE graph_lib)
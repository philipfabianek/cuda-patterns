#include <stdio.h>
#include <vector>
#include <random>
#include <chrono>
#include <queue>
#include "graph/graph.h"

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

constexpr int num_vertices = 100000;
constexpr int num_edges = 1000000;

constexpr int threads_per_block = 512;

/*
 * Performs BFS for a given level.
 - Finds vertices with distance equal to <prev_level> and iterates over their neighbors.
 - If the neighbor is unvisited, sets its distance to the current level (<prev_level> + 1).
 - Each thread is responsible for one vertex, one grid is launched per level.
 */
__global__ void bfs_kernel(const int *row_ptr, const int *col_indices, int *dist,
                           int *prev_frontier, int *prev_frontier_num,
                           int *curr_frontier, int *curr_frontier_num,
                           const int prev_level)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < *prev_frontier_num)
  {
    int u = prev_frontier[i];

    for (int j = row_ptr[u]; j < row_ptr[u + 1]; j++)
    {
      int v = col_indices[j];

      if (atomicCAS(&dist[v], -1, prev_level + 1) == -1)
      {
        int curr_frontier_idx = atomicAdd(curr_frontier_num, 1);
        curr_frontier[curr_frontier_idx] = v;
      }
    }
  }
}

int main()
{
  // Create random number generator and random distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  // Define host graph by first constructing a spanning tree
  // then adding random edges up to <num_edges>
  std::vector<Edge> edges = generate_random_spanning_tree(num_vertices, generator);
  add_random_edges(edges, num_vertices, num_edges, generator);

  // Convert the graph to the CSR format
  CSRGraph graph = convert_to_csr(edges, num_vertices);

  // Allocate memory for host output vector and initialize to -1,
  // set distance for the source node to 0
  std::vector<int> h_dist(num_vertices);
  std::fill(h_dist.begin(), h_dist.end(), -1);
  int source_vertex = 0;
  h_dist[source_vertex] = 0;

  // Prepare device variables
  size_t row_ptr_memsize = graph.row_ptr.size() * sizeof(int);
  size_t col_indices_memsize = graph.col_indices.size() * sizeof(int);
  size_t dist_memsize = graph.num_vertices * sizeof(int);
  size_t prev_frontier_memsize = graph.num_vertices * sizeof(int);
  size_t prev_frontier_num_memsize = sizeof(int);
  size_t curr_frontier_memsize = graph.num_vertices * sizeof(int);
  size_t curr_frontier_num_memsize = sizeof(int);

  int *d_row_ptr, *d_col_indices, *d_dist,
      *d_prev_frontier, *d_prev_frontier_num,
      *d_curr_frontier, *d_curr_frontier_num;

  CUDA_CHECK(cudaMalloc((void **)&d_row_ptr, row_ptr_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_col_indices, col_indices_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_dist, dist_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_prev_frontier, prev_frontier_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_prev_frontier_num, prev_frontier_num_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_curr_frontier, curr_frontier_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_curr_frontier_num, curr_frontier_num_memsize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_row_ptr, graph.row_ptr.data(), row_ptr_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_col_indices, graph.col_indices.data(), col_indices_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dist, h_dist.data(), dist_memsize, cudaMemcpyHostToDevice));

  // Set previous frontier
  int h_prev_frontier_num = 1;
  CUDA_CHECK(cudaMemcpy(d_prev_frontier_num, &h_prev_frontier_num, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prev_frontier, &source_vertex, sizeof(int), cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Run BFS and compute distances on GPU
  int prev_level = 0;
  while (h_prev_frontier_num > 0)
  {
    int h_curr_frontier_num = 0;
    CUDA_CHECK(cudaMemcpy(d_curr_frontier_num, &h_curr_frontier_num, sizeof(int), cudaMemcpyHostToDevice));

    int blocks_per_grid = (h_prev_frontier_num + threads_per_block - 1) / threads_per_block;
    bfs_kernel<<<blocks_per_grid, threads_per_block>>>(d_row_ptr, d_col_indices, d_dist,
                                                       d_prev_frontier, d_prev_frontier_num,
                                                       d_curr_frontier, d_curr_frontier_num,
                                                       prev_level);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_curr_frontier_num, d_curr_frontier_num, sizeof(int), cudaMemcpyDeviceToHost));

    int *temp_ptr_frontier = d_prev_frontier;
    d_prev_frontier = d_curr_frontier;
    d_curr_frontier = temp_ptr_frontier;

    int *temp_ptr_frontier_num = d_prev_frontier_num;
    d_prev_frontier_num = d_curr_frontier_num;
    d_curr_frontier_num = temp_ptr_frontier_num;

    h_prev_frontier_num = h_curr_frontier_num;

    prev_level++;
  }

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms (use NCU for a more precise measurement!)\n", milliseconds);

  // Copy data from device to host
  CUDA_CHECK(cudaMemcpy(h_dist.data(), d_dist, dist_memsize, cudaMemcpyDeviceToHost));

  // Verify the result and measure CPU execution time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  if (verify_bfs(graph, h_dist, source_vertex) != 0)
  {
    printf("Verification failed\n");
  }
  else
  {
    printf("All values match\n");
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
  printf("CPU execution time: %f milliseconds\n", cpu_time.count());

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_row_ptr));
  CUDA_CHECK(cudaFree(d_col_indices));
  CUDA_CHECK(cudaFree(d_dist));
  CUDA_CHECK(cudaFree(d_prev_frontier));
  CUDA_CHECK(cudaFree(d_prev_frontier_num));
  CUDA_CHECK(cudaFree(d_curr_frontier));
  CUDA_CHECK(cudaFree(d_curr_frontier_num));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
#include <stdio.h>
#include <vector>
#include <random>
#include <chrono>

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

constexpr dim3 grid_size(100, 100, 100);
constexpr int num_atoms = 100;
__constant__ float d_atom_data[4 * num_atoms];
// constexpr int samples_to_check = 10000;

constexpr dim3 threads_per_block(4, 4, 4);
constexpr int coarse_factor = 5;
constexpr int blocks_x = (grid_size.x + (threads_per_block.x * coarse_factor) - 1) / (threads_per_block.x * coarse_factor);
constexpr int blocks_y = (grid_size.y + threads_per_block.y - 1) / threads_per_block.y;
constexpr int blocks_z = (grid_size.z + threads_per_block.z - 1) / threads_per_block.z;
constexpr dim3 num_blocks(blocks_x, blocks_y, blocks_z);

/*
 * Calculates (electrostatic) potential for each grid point.
 * - Each thread computes the potential for <coarse_factor> grid points.
 * - Potential is calculated by iterating over all atoms and adding their contributions,
 *   which are their charges divided by the distance from the grid point.
 */
__global__ void potential_kernel(float *d_potential_map)
{
  int initial_x = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  float potential[coarse_factor] = {0};

  // clang-format off
  #pragma unroll
  for (int atom_idx = 0; atom_idx < num_atoms; atom_idx++)
  {
    float atom_x = d_atom_data[4 * atom_idx];
    float atom_y = d_atom_data[4 * atom_idx + 1];
    float atom_z = d_atom_data[4 * atom_idx + 2];
    float atom_charge = d_atom_data[4 * atom_idx + 3];

    float initial_dist_sq = (atom_y - (float)y) * (atom_y - (float)y) + (atom_z - (float)z) * (atom_z - (float)z) + 1e-12f;

    for (int i = 0; i < coarse_factor; i++)
    {
      int x = initial_x + i * blockDim.x;
      float dist_sq = initial_dist_sq + (atom_x - (float)x) * (atom_x - (float)x);
      potential[i] += atom_charge * rsqrtf(dist_sq);
    }
  }

  for (int i = 0; i < coarse_factor; i++)
  {
    int x = initial_x + i * blockDim.x;
    int index = z * grid_size.x * grid_size.y + y * grid_size.x + x;
    if (x < grid_size.x && y < grid_size.y && z < grid_size.z)
    {
      d_potential_map[index] = potential[i];
    }
  }
}

int verify_potential_map(const std::vector<float> &atom_data, const std::vector<float> &gpu_potential_map)
{
  // This can be used for larger grids and more atoms

  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::default_random_engine generator(seed);

  // std::uniform_int_distribution<> x_distribution(0, grid_size.x - 1);
  // std::uniform_int_distribution<> y_distribution(0, grid_size.y - 1);
  // std::uniform_int_distribution<> z_distribution(0, grid_size.z - 1);

  // for (int s = 0; s < samples_to_check; s++)
  // {
  //   int x = x_distribution(generator);
  //   int y = y_distribution(generator);
  //   int z = z_distribution(generator);
  // }

  for (int z = 0; z < grid_size.z; z++)
  {
    for (int y = 0; y < grid_size.y; y++)
    {
      for (int x = 0; x < grid_size.x; x++)
      {
        float potential = 0;

        for (int atom_idx = 0; atom_idx < num_atoms; atom_idx++)
        {
          float atom_x = atom_data[4 * atom_idx];
          float atom_y = atom_data[4 * atom_idx + 1];
          float atom_z = atom_data[4 * atom_idx + 2];
          float atom_charge = atom_data[4 * atom_idx + 3];
          float dist = sqrtf((atom_x - (float)x) * (atom_x - (float)x) + (atom_y - (float)y) * (atom_y - (float)y) + (atom_z - (float)z) * (atom_z - (float)z) + 1e-12f);
          potential += atom_charge / dist;
        }

        int index = z * grid_size.x * grid_size.y + y * grid_size.x + x;
        if (fabs(gpu_potential_map[index] - potential) > 1e-5f)
        {
          printf("Mismatch at index %d: expected %f, got %f\n", index, potential, gpu_potential_map[index]);
          return 1;
        }
      }
    }
  }

  return 0;
}

int main()
{
  // Create random number generator and random distributions
  // for coordinates and atom charges
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> x_distribution(0.0f, (float)(grid_size.x - 1));
  std::uniform_real_distribution<float> y_distribution(0.0f, (float)(grid_size.y - 1));
  std::uniform_real_distribution<float> z_distribution(0.0f, (float)(grid_size.z - 1));
  std::uniform_real_distribution<float> charge_distribution(-1.0f, 1.0f);

  // Define host atom data
  std::vector<float> h_atom_data(4 * num_atoms);

  for (int i = 0; i < num_atoms; i++)
  {
    h_atom_data[4 * i] = x_distribution(generator);
    h_atom_data[4 * i + 1] = y_distribution(generator);
    h_atom_data[4 * i + 2] = z_distribution(generator);
    h_atom_data[4 * i + 3] = charge_distribution(generator);
  }

  // Allocate memory for host potential map result
  std::vector<float> h_potential_map(grid_size.x * grid_size.y * grid_size.z, 0.0f);

  // Prepare device variables
  size_t potential_map_memsize = grid_size.x * grid_size.y * grid_size.z * sizeof(float);
  float *d_potential_map;
  CUDA_CHECK(cudaMalloc((void **)&d_potential_map, potential_map_memsize));

  // Copy atom data to constant memory
  size_t atom_data_memsize = 4 * num_atoms * sizeof(float);
  CUDA_CHECK(cudaMemcpyToSymbol(d_atom_data, h_atom_data.data(), atom_data_memsize));

  // Copy initial potential map data from host to device
  CUDA_CHECK(cudaMemcpy(d_potential_map, h_potential_map.data(), potential_map_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform potential map calculation on GPU
  potential_kernel<<<num_blocks, threads_per_block>>>(d_potential_map);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms (use NCU for a more precise measurement!)\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_potential_map.data(), d_potential_map, potential_map_memsize, cudaMemcpyDeviceToHost));

  // Verify output and measure CPU execution time
  auto cpu_start = std::chrono::high_resolution_clock::now();
  if (verify_potential_map(h_atom_data, h_potential_map) != 0)
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
  CUDA_CHECK(cudaFree(d_potential_map));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
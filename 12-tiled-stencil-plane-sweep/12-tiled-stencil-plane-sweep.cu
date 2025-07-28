#include <stdio.h>
#include <stdlib.h>
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

#define N 500
#define RANDOM_INITIALIZATION false
#define INPUT_TILE_SIZE 8
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - 2)
#define SAMPLES_TO_CHECK 100000

// Coefficients for the 7-point stencil (one for host and one for device)
const float h_c[7] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
__constant__ float d_c[7] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

/*
 * Performs a 3D stencil sweep on a 3D tensor using a 7-point stencil.
 * - Each thread block is responsible for one output tile.
 * - The threads compute elements along the z axis.
 * - Since the stencil radius is 1, it is sufficient to keep in memory values corresponding
 *   to 3 xy planes (previous, current, and next).
 * - Boundary elements are ignored in this case.
 */
__global__ void tiled_stencil_plane_sweep_kernel(float *d_A, float *d_B)
{
  int x = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - 1;
  int y = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - 1;
  int initial_z = blockIdx.z * OUTPUT_TILE_SIZE;

  // Keep values corresponding to the previous, current,
  // and next xy planes in shared memory
  __shared__ float prev[INPUT_TILE_SIZE][INPUT_TILE_SIZE];
  __shared__ float curr[INPUT_TILE_SIZE][INPUT_TILE_SIZE];
  __shared__ float next[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

  if (x >= 0 && x < N &&
      y >= 0 && y < N &&
      initial_z - 1 >= 0 && initial_z - 1 < N)
  {
    prev[threadIdx.y][threadIdx.x] = d_A[(initial_z - 1) * N * N + y * N + x];
  }

  if (x >= 0 && x < N &&
      y >= 0 && y < N &&
      initial_z < N)
  {
    curr[threadIdx.y][threadIdx.x] = d_A[initial_z * N * N + y * N + x];
  }

  // Iterate over the xy planes in the z direction
  for (int z = initial_z + 1; z <= initial_z + OUTPUT_TILE_SIZE; z++)
  {
    if (x >= 0 && x < N &&
        y >= 0 && y < N &&
        z < N)
    {
      next[threadIdx.y][threadIdx.x] = d_A[z * N * N + y * N + x];
    }

    __syncthreads();

    // Compute indices of the thread inside the output tile
    int output_tile_x = threadIdx.x - 1;
    int output_tile_y = threadIdx.y - 1;

    // Check whether the thread should be enabled for the current output tile
    // (threads on the edges are disabled because input tiles are bigger than output tiles)
    if (output_tile_x >= 0 && output_tile_x < OUTPUT_TILE_SIZE &&
        output_tile_y >= 0 && output_tile_y < OUTPUT_TILE_SIZE)
    {
      if (x >= 0 && x < N &&
          y >= 0 && y < N &&
          z < N)
      {
        // Approximate inner values using a 7-point stencil
        float sum = d_c[0] * curr[threadIdx.y][threadIdx.x] +
                    d_c[1] * prev[threadIdx.y][threadIdx.x] +
                    d_c[2] * next[threadIdx.y][threadIdx.x] +
                    d_c[3] * curr[threadIdx.y - 1][threadIdx.x] +
                    d_c[4] * curr[threadIdx.y + 1][threadIdx.x] +
                    d_c[5] * curr[threadIdx.y][threadIdx.x - 1] +
                    d_c[6] * curr[threadIdx.y][threadIdx.x + 1];
        d_B[(z - 1) * N * N + y * N + x] = sum;
      }
    }

    __syncthreads();

    // Shift the shared memory buffers for the next iteration
    prev[threadIdx.y][threadIdx.x] = curr[threadIdx.y][threadIdx.x];
    curr[threadIdx.y][threadIdx.x] = next[threadIdx.y][threadIdx.x];
  }
}

int verify_stencil_sweep(float *h_A, float *h_B)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<> x_dist(0, N - 1);
  std::uniform_int_distribution<> y_dist(0, N - 1);
  std::uniform_int_distribution<> z_dist(0, N - 1);

  for (int s = 0; s < SAMPLES_TO_CHECK; s++)
  {
    int i = x_dist(generator);
    int j = y_dist(generator);
    int k = z_dist(generator);

    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
      float target_value = h_B[k * N * N + j * N + i];
      float expected_value = h_c[0] * h_A[k * N * N + j * N + i] +
                             h_c[1] * h_A[(k - 1) * N * N + j * N + i] +
                             h_c[2] * h_A[(k + 1) * N * N + j * N + i] +
                             h_c[3] * h_A[k * N * N + (j - 1) * N + i] +
                             h_c[4] * h_A[k * N * N + (j + 1) * N + i] +
                             h_c[5] * h_A[k * N * N + j * N + (i - 1)] +
                             h_c[6] * h_A[k * N * N + j * N + (i + 1)];

      if (fabs(target_value - expected_value) > 1e-5)
      {
        printf("Mismatch (%d, %d, %d): expected %f, got %f\n", i, j, k, expected_value, target_value);
        return 1;
      }
    }
  }

  return 0;
}

int main()
{
  // Create random number generator and random distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

  // Define 3D data tensor
  size_t A_memsize = N * N * N * sizeof(float);
  float *h_A = (float *)malloc(A_memsize);

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        if (RANDOM_INITIALIZATION)
        {
          h_A[i * N * N + j * N + k] = distribution(generator);
        }
        else
        {
          h_A[i * N * N + j * N + k] = 1.0f;
        }
      }
    }
  }

  // Allocate memory for tensor B which will be the result
  // of applying the stencil kernel to the tensor A
  size_t B_memsize = A_memsize;
  float *h_B = (float *)malloc(B_memsize);

  // Prepare device variables for the tensors
  float *d_A, *d_B;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, A_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform stencil sweep on GPU
  dim3 threads_per_block(INPUT_TILE_SIZE, INPUT_TILE_SIZE, 1);
  int blocks_x = (N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE;
  int blocks_y = (N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE;
  int blocks_z = (N + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE;
  dim3 num_blocks(blocks_x, blocks_y, blocks_z);
  tiled_stencil_plane_sweep_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_B, d_B, B_memsize, cudaMemcpyDeviceToHost));

  // Check values
  if (verify_stencil_sweep(h_A, h_B) != 0)
  {
    return 1;
  }
  printf("All (sampled) values match\n");

  // Free memory and destroy events
  free(h_A);
  free(h_B);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
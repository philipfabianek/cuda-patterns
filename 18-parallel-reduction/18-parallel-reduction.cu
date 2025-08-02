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

constexpr int N = 100 * 1000 * 1000;
constexpr int threads_per_block = 256;
constexpr int coarse_factor = 16;
constexpr int blocks_per_grid = (N + (coarse_factor * threads_per_block) - 1) /
                                (coarse_factor * threads_per_block);

/*
 * Reduces an array of integers w.r.t. the addition operation.
 * - Each thread loads <coarse_factor> input values before storing
 *   their sum into the shared memory.
 * - The values in shared memory are then reduced using a decreasing stride,
 *   as the stride decreases the number of inactive threads increases.
 * - The final result stored in temp[0] is then added to the global
 *   sum using an atomic operation to prevent race conditions.
 */
__global__ void reduction_kernel(unsigned int *d_input_array, unsigned int *d_sum_result)
{
  int base_idx = coarse_factor * blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ unsigned int temp[threads_per_block];

  // Load elements from the input array while computing their sum
  unsigned int sum = 0;

  for (int i = 0; i < coarse_factor; i++)
  {
    if (base_idx + i * blockDim.x < N)
    {
      sum += d_input_array[base_idx + i * blockDim.x];
    }
  }

  temp[threadIdx.x] = sum;

  // Perform reduction in shared memory
  for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
  {
    __syncthreads();

    if (threadIdx.x < stride)
    {
      temp[threadIdx.x] += temp[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0)
  {
    atomicAdd(d_sum_result, temp[0]);
  }
}

int verify_sum(const unsigned int *h_input_array, const unsigned int *h_gpu_sum)
{
  unsigned int target_sum = 0;

  for (int i = 0; i < N; ++i)
  {
    target_sum += h_input_array[i];
  }

  if (target_sum != *h_gpu_sum)
  {
    printf("Mismatch: expected sum %d, got %d\n", target_sum, *h_gpu_sum);
    return 1;
  }

  return 0;
}

int main()
{
  // Create random number generator and random distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_int_distribution<unsigned int> distribution(0, 4);

  // Define input array with values from the random distribution
  size_t input_memsize = N * sizeof(unsigned int);
  unsigned int *h_input_array = (unsigned int *)malloc(input_memsize);

  for (int i = 0; i < N; ++i)
  {
    h_input_array[i] = distribution(generator);
  }

  // Allocate memory for the host sum result
  size_t sum_memsize = sizeof(unsigned int);
  unsigned int *h_sum_result = (unsigned int *)malloc(sum_memsize);

  // Prepare device variables
  unsigned int *d_input_array;
  unsigned int *d_sum_result;
  CUDA_CHECK(cudaMalloc((void **)&d_input_array, input_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_sum_result, sum_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_array, h_input_array, input_memsize, cudaMemcpyHostToDevice));

  // Initialize the sum memory on the device to zero
  CUDA_CHECK(cudaMemset(d_sum_result, 0, sum_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform reduction on GPU
  reduction_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_array, d_sum_result);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_sum_result, d_sum_result, sum_memsize, cudaMemcpyDeviceToHost));

  // Check values
  printf("Verifying sum...\n");
  if (verify_sum(h_input_array, h_sum_result) != 0)
  {
    return 1;
  }
  printf("All values match\n");

  // Free memory and destroy events
  free(h_input_array);
  free(h_sum_result);

  CUDA_CHECK(cudaFree(d_input_array));
  CUDA_CHECK(cudaFree(d_sum_result));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
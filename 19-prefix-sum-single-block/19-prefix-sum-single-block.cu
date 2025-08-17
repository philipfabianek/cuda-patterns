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

// This program works with only one block, so only up to 1024 threads.
constexpr int N = 1024;

constexpr int threads_per_block = 1024;
constexpr int blocks_per_grid = 1;

/*
 * Kogge-Stone prefix sum algorithm.
 */
__global__ void prefix_sum_kernel(unsigned int *d_input_array, unsigned int *d_sum_result)
{
  __shared__ unsigned int temp_result[N];
  unsigned int tx = threadIdx.x;
  temp_result[tx] = d_input_array[tx];

  unsigned int temp;
  for (int stride = 1; stride < N; stride *= 2)
  {
    __syncthreads();

    if (tx >= stride)
    {
      temp = temp_result[tx - stride] + temp_result[tx];
    }

    __syncthreads();

    if (tx >= stride)
    {
      temp_result[tx] = temp;
    }
  }

  d_sum_result[tx] = temp_result[tx];
}

// /*
//  * Brent-Kung prefix sum algorithm.
//  */
// __global__ void prefix_sum_kernel(unsigned int *d_input_array, unsigned int *d_sum_result)
// {
//   __shared__ unsigned int temp_result[N];
//   unsigned int tx = threadIdx.x;
//   temp_result[tx] = d_input_array[tx];

//   for (int stride = 1; stride < N; stride *= 2)
//   {
//     __syncthreads();

//     int idx = 2 * stride * (tx + 1) - 1;
//     if (idx < N)
//     {
//       temp_result[idx] += temp_result[idx - stride];
//     }
//   }

//   for (int stride = N / (2 * 2); stride >= 1; stride /= 2)
//   {
//     __syncthreads();

//     int idx = 2 * stride * (tx + 1) - 1;
//     if (idx + stride < N)
//     {
//       temp_result[idx + stride] += temp_result[idx];
//     }
//   }

//   __syncthreads();

//   d_sum_result[tx] = temp_result[tx];
// }

int verify_prefix_sum(const unsigned int *h_input_array, const unsigned int *h_gpu_sum)
{
  std::vector<unsigned int> target_sum(N);

  clock_t start_time = clock();

  target_sum[0] = h_input_array[0];

  for (int i = 1; i < N; ++i)
  {
    target_sum[i] = target_sum[i - 1] + h_input_array[i];
  }

  clock_t end_time = clock();
  double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
  printf("CPU verification time: %f seconds\n", elapsed_time);

  for (int i = 0; i < N; ++i)
  {
    if (target_sum[i] != h_gpu_sum[i])
    {
      printf("Mismatch at index %d: expected %d, got %d\n", i, target_sum[i], h_gpu_sum[i]);
      return 1;
    }
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
  std::vector<unsigned int> h_input_array(N);

  for (int i = 0; i < N; ++i)
  {
    h_input_array[i] = distribution(generator);
  }

  // Allocate memory for the host prefix sum result
  size_t sum_memsize = N * sizeof(unsigned int);
  std::vector<unsigned int> h_sum_result(N);

  // Prepare device variables
  unsigned int *d_input_array;
  unsigned int *d_sum_result;
  CUDA_CHECK(cudaMalloc((void **)&d_input_array, input_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_sum_result, sum_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_array, h_input_array.data(), input_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Compute prefix sum on GPU
  prefix_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_array, d_sum_result);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_sum_result.data(), d_sum_result, sum_memsize, cudaMemcpyDeviceToHost));

  // Check values and measure execution time
  printf("Verifying sum...\n");
  if (verify_prefix_sum(h_input_array.data(), h_sum_result.data()) != 0)
  {
    return 1;
  }
  printf("All values match\n");

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_input_array));
  CUDA_CHECK(cudaFree(d_sum_result));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
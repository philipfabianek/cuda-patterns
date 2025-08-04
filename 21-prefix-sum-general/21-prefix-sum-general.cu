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

// This configuration yielded the best performance on my GPU (RTX 3070)
constexpr unsigned int threads_per_block = 128;
constexpr unsigned int coarse_factor = 64;
constexpr unsigned int base_block_size = threads_per_block * coarse_factor;
constexpr unsigned int N = 100 * 1000 * 1000;

/*
 * Performs one level of the "up-sweep" phase of the scan.
 * - A block always loads a chunk of data into shared memory,
 *   the stride used to access global memory depends on the level.
 * - At level 0, it reads from the original input array in a coalesced manner.
 * - At higher levels, it reads the block sums from the previous levels,
 *   which are always stored in the last element of each block.
 * - Note that after level 0 everything happens in-place.
 */
__global__ void upsweep_kernel(unsigned int *d_input_array, unsigned int *d_sum_result, unsigned int level, unsigned int level_stride)
{
  unsigned int block_start = blockIdx.x * base_block_size;
  unsigned int tx = threadIdx.x;
  unsigned int block_sum_idx = (tx + 1) * coarse_factor - 1;
  int prev_block_sum_idx = block_sum_idx - coarse_factor;

  __shared__ unsigned int block_data[base_block_size];

  // Load data into shared memory, this is coalesced
  // for level 0 which is the level that takes by far the most time
  for (unsigned int i = 0; i < coarse_factor; i++)
  {
    unsigned int smem_idx = i * blockDim.x + tx;
    long long gmem_idx = (long long)(block_start + smem_idx + 1) * level_stride - 1;
    if (gmem_idx < N)
    {
      if (level == 0)
      {
        block_data[smem_idx] = d_input_array[gmem_idx];
      }
      else
      {
        block_data[smem_idx] = d_sum_result[gmem_idx];
      }
    }
  }

  __syncthreads();

  // Perform prefix sum on each chunk (= <coarse_factor>
  // contiguous elements inside the shared memory for each thread),
  // this finishes the per-thread sequential scan
  for (unsigned int i = 1; i < coarse_factor; i++)
  {
    unsigned int smem_idx = tx * coarse_factor + i;
    block_data[smem_idx] += block_data[smem_idx - 1];
  }

  // Perform the Kogge-Stone prefix sum algorithm
  // on last elements of each chunk.
  unsigned int temp;
  for (unsigned int stride = coarse_factor; stride < base_block_size; stride *= 2)
  {
    __syncthreads();

    if (block_sum_idx >= stride)
    {
      temp = block_data[block_sum_idx - stride] + block_data[block_sum_idx];
    }

    __syncthreads();

    if (block_sum_idx >= stride)
    {
      block_data[block_sum_idx] = temp;
    }
  }

  __syncthreads();

  // Add the last element from the previous chunk
  // to all but the last element in the current chunk
  if (prev_block_sum_idx >= 0)
  {
    for (unsigned int i = 0; i < coarse_factor - 1; i++)
    {
      unsigned int smem_idx = tx * coarse_factor + i;
      block_data[smem_idx] += block_data[prev_block_sum_idx];
    }
  }

  __syncthreads();

  // Write the results back to the global memory
  for (unsigned int i = 0; i < coarse_factor; i++)
  {
    unsigned int smem_idx = i * blockDim.x + tx;
    long long gmem_idx = (long long)(block_start + smem_idx + 1) * level_stride - 1;
    if (gmem_idx < N)
    {
      d_sum_result[gmem_idx] = block_data[smem_idx];
    }
  }
}

/*
 * Performs one level of the "down-sweep" phase of the scan.
 * - This phase adds partial sums (offsets) to create
 *   the final prefix sum for all elements.
 * - Each block is responsible for adding a single offset value from the previous
 *   higher level to all but the last element in the current level.
 */
__global__ void downsweep_kernel(unsigned int *d_sum_result, unsigned int level_stride, unsigned int level_block_size)
{
  unsigned int block_start = blockIdx.x * base_block_size;
  unsigned int tx = threadIdx.x;

  // This is the offset value that needs to be added in this level
  int block_offset_gmem_idx = blockIdx.x * level_block_size - 1;
  if (block_offset_gmem_idx >= 0)
  {
    unsigned int block_offset_value = d_sum_result[block_offset_gmem_idx];
    for (unsigned int i = 0; i < coarse_factor; i++)
    {
      unsigned int smem_idx = i * blockDim.x + tx;
      long long gmem_idx = (block_start + smem_idx + 1) * level_stride - 1;
      if (smem_idx != base_block_size - 1 && gmem_idx < N)
      {
        d_sum_result[gmem_idx] += block_offset_value;
      }
    }
  }
}

int verify_prefix_sum(const unsigned int *h_input_array, const unsigned int *h_gpu_sum)
{
  size_t input_memsize = N * sizeof(unsigned int);
  unsigned int *target_sum = (unsigned int *)malloc(input_memsize);

  clock_t start_time = clock();

  target_sum[0] = h_input_array[0];

  for (size_t i = 1; i < N; i++)
  {
    target_sum[i] = target_sum[i - 1] + h_input_array[i];
  }

  clock_t end_time = clock();
  double elapsed_time = double(end_time - start_time) / CLOCKS_PER_SEC;
  printf("CPU verification time: %f seconds\n", elapsed_time);

  for (int i = 0; i < N; i++)
  {
    if (target_sum[i] != h_gpu_sum[i])
    {
      printf("Mismatch at index %d: expected %d, got %d\n", i, target_sum[i], h_gpu_sum[i]);
      free(target_sum);
      return 1;
    }
  }

  free(target_sum);

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

  for (size_t i = 0; i < N; i++)
  {
    h_input_array[i] = distribution(generator);
  }

  // Allocate memory for the host prefix sum result
  size_t sum_memsize = N * sizeof(unsigned int);
  unsigned int *h_sum_result = (unsigned int *)malloc(sum_memsize);

  // Prepare device variables
  unsigned int *d_input_array;
  unsigned int *d_sum_result;
  CUDA_CHECK(cudaMalloc((void **)&d_input_array, input_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_sum_result, sum_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_array, h_input_array, input_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Compute prefix sum on GPU
  unsigned int max_level = 0;
  long long max_block_size = base_block_size;
  while ((N - 1) / max_block_size >= 1)
  {
    max_level += 1;
    max_block_size *= base_block_size;
  }

  // Compute prefix sums at increasing levels to cover the entire array,
  // levels greater than 0 always work with the last elements
  // of each block from the previous level
  long long level_stride = 1;
  for (int level = 0; level <= max_level; level += 1)
  {
    long long level_block_size = level_stride * base_block_size;
    unsigned int blocks_per_grid = (N + level_block_size - 1) / level_block_size;
    upsweep_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_array, d_sum_result, level, level_stride);
    CUDA_CHECK(cudaGetLastError());
    level_stride *= base_block_size;
  }

  // Perform downsweep to add the offsets to the elements that need them
  level_stride /= base_block_size * base_block_size;
  for (int level = max_level - 1; level >= 0; level -= 1)
  {
    long long level_block_size = level_stride * base_block_size;
    unsigned int blocks_per_grid = (N + level_block_size - 1) / level_block_size;
    downsweep_kernel<<<blocks_per_grid, threads_per_block>>>(d_sum_result, level_stride, level_block_size);
    CUDA_CHECK(cudaGetLastError());
    level_stride /= base_block_size;
  }

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_sum_result, d_sum_result, sum_memsize, cudaMemcpyDeviceToHost));

  // Check values and measure execution time
  printf("Verifying sum...\n");
  if (verify_prefix_sum(h_input_array, h_sum_result) != 0)
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
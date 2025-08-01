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

constexpr int N = 1000 * 1000 * 1000;
constexpr char initial_char = 'A';
constexpr int num_bins = 64;
// Cannot be too large due because of signed char
// ranging from -128 to 127 (A is 65)
constexpr int num_unique_chars = 50;

// These parameters were optimized using grid search (see run_search.sh),
// this of course depends on the GPU (mine is RTX 3070)
constexpr int threads_per_block = 512;
constexpr int coarse_factor = 16;
constexpr int block_multiplier = 16;

/*
 * Computes a histogram of character frequencies.
 * - Each thread processes <coarse_factor> contiguous input characters
 *   called a "chunk" before jumping to the next chunk in a grid-stride loop.
 * - Histograms for each block are stored in shared memory
 *   before being merged into the final histogram.
 */
__global__ void histogram_kernel(const char *d_input_string, unsigned int *d_histogram)
{
  __shared__ unsigned int shared_histogram[num_bins];

  // Using this loop (and the loop at the end of the kernel)
  // handles cases where the number of bins would be greater than the number of threads
  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
  {
    shared_histogram[i] = 0;
  }

  __syncthreads();

  for (int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
       chunk_idx < (N + coarse_factor - 1) / coarse_factor;
       chunk_idx += gridDim.x * blockDim.x)
  {
    int base_idx = chunk_idx * coarse_factor;

    for (int j = 0; j < coarse_factor; j++)
    {
      int idx = base_idx + j;
      if (idx < N)
      {
        char current_char = d_input_string[idx];
        unsigned int destination_bin = (current_char - initial_char) * num_bins / num_unique_chars;
        atomicAdd(&shared_histogram[destination_bin], 1);
      }
    }
  }

  __syncthreads();

  // Merge the shared histogram into the final histogram
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
  {
    unsigned int bin_value = shared_histogram[bin];
    atomicAdd(&d_histogram[bin], bin_value);
  }
}

int verify_histogram(const char *h_input_string, const unsigned int *h_histogram)
{
  unsigned int target_histogram[num_bins] = {0};

  for (int i = 0; i < N; i++)
  {
    char current_char = h_input_string[i];
    unsigned int bin_index = (current_char - initial_char) * num_bins / num_unique_chars;
    target_histogram[bin_index]++;
  }

  for (int i = 0; i < num_bins; i++)
  {
    if (target_histogram[i] != h_histogram[i])
    {
      printf("Mismatch at bin %d: expected %u, got %u\n", i, target_histogram[i], h_histogram[i]);
      return 1;
    }
  }

  return 0;
}

int main(int argc, char *argv[])
{
  // This code was used during the grid search
  // to find the best hyperparameters (see run_search.sh )

  // if (argc != 4)
  // {
  //   fprintf(stderr, "Usage: %s <threads_per_block> <coarse_factor> <block_multiplier>\n", argv[0]);
  //   return 1;
  // }

  // const int threads_per_block = atoi(argv[1]);
  // const int coarse_factor = atoi(argv[2]);
  // const int block_multiplier = atoi(argv[3]);

  // Obtain device properties, the number of blocks will be
  // a multiple of the number of streaming multiprocessors
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));

  // Define input string
  size_t input_memsize = N * sizeof(char);
  char *h_input_string = (char *)malloc(input_memsize);

  // Make 90% characters the same to make the optimization more challenging,
  // use all characters for the last 10%
  for (int i = 0; i < N; i++)
  {
    if (i < 0.9 * N)
    {
      h_input_string[i] = initial_char;
    }
    else
    {
      h_input_string[i] = initial_char + (i % (num_unique_chars - 1)) + 1;
    }
  }

  // Allocate memory for the host histogram result
  size_t histogram_memsize = num_bins * sizeof(unsigned int);
  unsigned int *h_histogram = (unsigned int *)malloc(histogram_memsize);

  // Prepare device variables
  char *d_input_string;
  unsigned int *d_histogram;
  CUDA_CHECK(cudaMalloc((void **)&d_input_string, input_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_histogram, histogram_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_string, h_input_string, input_memsize, cudaMemcpyHostToDevice));

  // Initialize the histogram memory on the device to all zeros
  CUDA_CHECK(cudaMemset(d_histogram, 0, histogram_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform histogram calculation on GPU
  const int blocks_per_grid = block_multiplier * props.multiProcessorCount;
  histogram_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_string, d_histogram);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, histogram_memsize, cudaMemcpyDeviceToHost));

  // Check values
  printf("Verifying histogram...\n");
  if (verify_histogram(h_input_string, h_histogram) != 0)
  {
    return 1;
  }
  printf("All values match\n");

  // Free memory and destroy events
  free(h_input_string);
  free(h_histogram);

  CUDA_CHECK(cudaFree(d_input_string));
  CUDA_CHECK(cudaFree(d_histogram));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
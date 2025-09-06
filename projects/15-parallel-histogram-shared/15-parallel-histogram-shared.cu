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

constexpr int N = 512 * 512 * 512;
constexpr char initial_char = 'A';
constexpr int num_bins = 64;
// Cannot be too large due because of signed char
// ranging from -128 to 127 (A is 65)
constexpr int num_unique_chars = 50;

// Using 512 threads per block leads to the best performance on my GPU,
// it is slightly better than 256 threads per block
// and significantly better than 1024 threads per block
constexpr int threads_per_block = 512;
constexpr int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

/*
 * Computes a histogram of character frequencies.
 * - Each thread processes one input character.
 * - Histograms for each block are stored in shared memory
 *   before being merged into the final histogram, this reduces the number
 *   of overlapping atomic operations and significantly improves the performance.
 * - The performance increase compared to the previous privatized version
 *   is also very significant.
 */
__global__ void histogram_kernel(const char *d_input_string, unsigned int *d_histogram)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ unsigned int shared_histogram[num_bins];

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
  {
    shared_histogram[i] = 0;
  }

  __syncthreads();

  // Compute histogram for the current block
  if (idx < N)
  {
    char current_char = d_input_string[idx];
    unsigned int destination_bin = (current_char - initial_char) * num_bins / num_unique_chars;
    atomicAdd(&shared_histogram[destination_bin], 1);
  }

  __syncthreads();

  // Merge the shared histogram into the final histogram,
  // using this loop handles cases where the number of bins
  // would be greater than the number of threads
  for (int bin = threadIdx.x; bin < num_bins; bin += blockDim.x)
  {
    atomicAdd(&d_histogram[bin], shared_histogram[bin]);
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

int main()
{
  // Define input string
  size_t input_memsize = N * sizeof(char);
  std::vector<char> h_input_string(N);

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
  std::vector<unsigned int> h_histogram(num_bins);

  // Prepare device variables
  char *d_input_string;
  unsigned int *d_histogram;
  CUDA_CHECK(cudaMalloc((void **)&d_input_string, input_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_histogram, histogram_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_string, h_input_string.data(), input_memsize, cudaMemcpyHostToDevice));

  // Initialize the histogram memory on the device to all zeros
  CUDA_CHECK(cudaMemset(d_histogram, 0, histogram_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform histogram calculation on GPU
  histogram_kernel<<<blocks_per_grid, threads_per_block>>>(d_input_string, d_histogram);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms (use NCU for a more precise measurement!)\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram, histogram_memsize, cudaMemcpyDeviceToHost));

  // Check values
  printf("Verifying histogram...\n");
  if (verify_histogram(h_input_string.data(), h_histogram.data()) != 0)
  {
    return 1;
  }
  printf("All values match\n");

  // Free device memory
  CUDA_CHECK(cudaFree(d_input_string));
  CUDA_CHECK(cudaFree(d_histogram));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
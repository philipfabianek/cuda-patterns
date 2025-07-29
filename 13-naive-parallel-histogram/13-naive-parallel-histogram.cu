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

constexpr char INITIAL_CHAR = 'A';
constexpr int N = 1000 * 1000 * 1000;
constexpr int NUM_BINS = 64;
// Cannot be too large due because of signed char
// ranging from -128 to 127 (A is 65)
constexpr int NUM_UNIQUE_CHARS = 50;

/*
 * Computes a histogram of character frequencies.
 * - Each thread processes one input character.
 * - Race conditions are prevented by using atomicAdd().
 */
__global__ void histogram_kernel(const char *d_input_string, unsigned int *d_histogram)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N)
  {
    char current_char = d_input_string[idx];
    unsigned int bin_index = (current_char - INITIAL_CHAR) * NUM_BINS / NUM_UNIQUE_CHARS;
    atomicAdd(&d_histogram[bin_index], 1);
  }
}

int verify_histogram(const char *h_input_string, const unsigned int *h_histogram)
{
  unsigned int target_histogram[NUM_BINS] = {0};

  for (int i = 0; i < N; i++)
  {
    char current_char = h_input_string[i];
    unsigned int bin_index = (current_char - INITIAL_CHAR) * NUM_BINS / NUM_UNIQUE_CHARS;
    target_histogram[bin_index]++;
  }

  for (int i = 0; i < NUM_BINS; i++)
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
  char *h_input_string = (char *)malloc(input_memsize);

  // Make 90% characters the same to make the performance
  // roughly 3-4 times worse, use all characters for the last 10%
  for (int i = 0; i < N; i++)
  {
    if (i < 0.9 * N)
    {
      h_input_string[i] = INITIAL_CHAR;
    }
    else
    {
      h_input_string[i] = INITIAL_CHAR + (i % (NUM_UNIQUE_CHARS - 1)) + 1;
    }
  }

  // Allocate memory for the host histogram result
  size_t histogram_memsize = NUM_BINS * sizeof(unsigned int);
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
  int threads_per_block = 256;
  int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
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
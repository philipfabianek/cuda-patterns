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
#define BLOCK_SIZE 8
#define SAMPLES_TO_CHECK 100000

/*
 * Performs a 3D stencil sweep on a 3D tensor using a 7-point stencil.
 * - Each thread computes one element of the output tensor.
 * - Boundary elements are set to zero.
 */
__global__ void stencil_kernel(float *d_A, float *d_B)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 1 &&
      x < N - 1 &&
      y >= 1 &&
      y < N - 1 &&
      z >= 1 &&
      z < N - 1)
  {
    // Approximate inner values using a 7-point stencil
    float sum = d_A[z * N * N + y * N + x] +
                d_A[(z - 1) * N * N + y * N + x] +
                d_A[(z + 1) * N * N + y * N + x] +
                d_A[z * N * N + (y - 1) * N + x] +
                d_A[z * N * N + (y + 1) * N + x] +
                d_A[z * N * N + y * N + (x - 1)] +
                d_A[z * N * N + y * N + (x + 1)];
    d_B[z * N * N + y * N + x] = sum;
  }
  else
  {
    // Set boundary values to zero
    if (x < N && y < N && z < N)
    {
      d_B[z * N * N + y * N + x] = 0.0f;
    }
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

    float target_value = h_B[k * N * N + j * N + i];
    float expected_value = 0.0f;

    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
      expected_value = h_A[k * N * N + j * N + i] +
                       h_A[(k - 1) * N * N + j * N + i] +
                       h_A[(k + 1) * N * N + j * N + i] +
                       h_A[k * N * N + (j - 1) * N + i] +
                       h_A[k * N * N + (j + 1) * N + i] +
                       h_A[k * N * N + j * N + (i - 1)] +
                       h_A[k * N * N + j * N + (i + 1)];
    }

    if (fabs(target_value - expected_value) > 1e-5)
    {
      printf("Mismatch (%d, %d): expected %f, got %f\n", i, j, expected_value, target_value);
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
  dim3 threads_per_block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  int blocks_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blocks_y = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int blocks_z = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 num_blocks(blocks_x, blocks_y, blocks_z);
  stencil_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B);
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
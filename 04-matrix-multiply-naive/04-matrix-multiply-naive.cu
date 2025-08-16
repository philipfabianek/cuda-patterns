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

constexpr bool random_initialization = false;
constexpr int A_rows = 10000;
constexpr int A_cols = 10000;
constexpr int B_rows = 10000;
constexpr int B_cols = 10000;
constexpr int C_rows = A_rows;
constexpr int C_cols = B_cols;
constexpr int samples_to_check = 10000;

__global__ void naive_matrix_multiply(float *d_A, float *d_B, float *d_C)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < A_rows && col < B_cols)
  {
    float sum = 0;
    for (int k = 0; k < A_cols; k++)
    {
      sum += d_A[row * A_cols + k] * d_B[k * B_cols + col];
    }
    d_C[row * B_cols + col] = sum;
  }
}

int verify_matrix_multiplication(float *h_A, float *h_B, float *h_C)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<> row_dist(0, A_rows - 1);
  std::uniform_int_distribution<> col_dist(0, B_cols - 1);

  for (int s = 0; s < samples_to_check; s++)
  {
    int i = row_dist(generator);
    int j = col_dist(generator);

    float target_value = h_C[i * B_cols + j];
    float expected_value = 0.0f;

    for (int k = 0; k < A_cols; k++)
    {
      expected_value += h_A[i * A_cols + k] * h_B[k * B_cols + j];
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

  // Define matrix A with values from a random distribution
  size_t A_memsize = A_rows * A_cols * sizeof(float);
  float *h_A = (float *)malloc(A_memsize);

  for (int i = 0; i < A_rows; i++)
  {
    for (int j = 0; j < A_cols; j++)
    {
      if (random_initialization)
      {
        h_A[i * A_cols + j] = distribution(generator);
      }
      else
      {
        h_A[i * A_cols + j] = 1.0f;
      }
    }
  }

  // Define matrix B values from a random distribution
  size_t B_memsize = B_rows * B_cols * sizeof(float);
  float *h_B = (float *)malloc(B_memsize);

  for (int i = 0; i < B_rows; i++)
  {
    for (int j = 0; j < B_cols; j++)
    {
      if (random_initialization)
      {
        h_B[i * B_cols + j] = distribution(generator);
      }
      else
      {
        h_B[i * B_cols + j] = 1.0f;
      }
    }
  }

  // Allocate memory for matrix C which will be the product of A and B
  size_t C_memsize = C_rows * C_cols * sizeof(float);
  float *h_C = (float *)malloc(C_memsize);

  // Prepare device variables for the matrices
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_C, C_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, A_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, B_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform matrix multiplication on GPU
  dim3 threads_per_block(16, 16);
  int blocks_x = (C_cols + threads_per_block.x - 1) / threads_per_block.x;
  int blocks_y = (C_rows + threads_per_block.y - 1) / threads_per_block.y;
  dim3 num_blocks(blocks_x, blocks_y);
  naive_matrix_multiply<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_C, d_C, C_memsize, cudaMemcpyDeviceToHost));

  // Check values
  if (verify_matrix_multiplication(h_A, h_B, h_C) != 0)
  {
    return 1;
  }
  printf("All (sampled) values match\n");

  // Free memory and destroy events
  free(h_A);
  free(h_B);
  free(h_C);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
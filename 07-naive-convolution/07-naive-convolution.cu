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

constexpr int samples_to_check = 10000;

__global__ void naive_convolution(float *d_A, float *d_B, float *d_F, int matrix_rows, int matrix_cols, int filter_rows, int filter_cols, int filter_radius_y, int filter_radius_x)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < matrix_rows && col < matrix_cols)
  {
    float sum = 0;

    for (int i = -filter_radius_y; i < filter_radius_y + 1; i++)
    {
      for (int j = -filter_radius_x; j < filter_radius_x + 1; j++)
      {
        if (row + i >= 0 && row + i < matrix_rows && col + j >= 0 && col + j < matrix_cols)
        {
          sum += d_A[(row + i) * matrix_cols + (col + j)] * d_F[(filter_radius_y + i) * filter_cols + (filter_radius_x + j)];
        }
      }
    }

    d_B[row * matrix_cols + col] = sum;
  }
}

int verify_convolution(float *h_A, float *h_B, float *h_F, int matrix_rows, int matrix_cols, int filter_rows, int filter_cols, int filter_radius_y, int filter_radius_x)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<> row_dist(0, matrix_rows - 1);
  std::uniform_int_distribution<> col_dist(0, matrix_cols - 1);

  for (int s = 0; s < samples_to_check; s++)
  {
    int i = row_dist(generator);
    int j = col_dist(generator);

    float target_value = h_B[i * matrix_cols + j];
    float expected_value = 0.0f;

    for (int k = -filter_radius_y; k < filter_radius_y + 1; k++)
    {
      for (int l = -filter_radius_x; l < filter_radius_x + 1; l++)
      {
        if (i + k >= 0 && i + k < matrix_rows && j + l >= 0 && j + l < matrix_cols)
        {
          expected_value += h_A[(i + k) * matrix_cols + (j + l)] * h_F[(filter_radius_y + k) * filter_cols + (filter_radius_x + l)];
        }
      }
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
  int matrix_rows = 3000;
  int matrix_cols = 2000;
  size_t A_memsize = matrix_rows * matrix_cols * sizeof(float);
  float *h_A = (float *)malloc(A_memsize);

  for (int i = 0; i < matrix_rows; i++)
  {
    for (int j = 0; j < matrix_cols; j++)
    {
      h_A[i * matrix_cols + j] = distribution(generator);
    }
  }

  // Define filter F with values from a random distribution
  int filter_rows = 7;
  int filter_cols = 5;
  size_t F_memsize = filter_rows * filter_cols * sizeof(float);
  float *h_F = (float *)malloc(F_memsize);

  for (int i = 0; i < filter_rows; i++)
  {
    for (int j = 0; j < filter_cols; j++)
    {
      h_F[i * filter_cols + j] = distribution(generator);
    }
  }

  // Allocate memory for matrix B which will be the result of the convolution
  // (formally cross-correlation) of the matrix A with the filter F
  // (it will have same size as A, zeros will be used as padding)
  size_t B_memsize = A_memsize;
  float *h_B = (float *)malloc(B_memsize);

  // Prepare device variables for the matrices and the filter
  float *d_A, *d_B, *d_F;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_F, F_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, A_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_F, h_F, F_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform convolution on GPU
  dim3 threads_per_block(16, 16);
  int blocks_x = (matrix_cols + threads_per_block.x - 1) / threads_per_block.x;
  int blocks_y = (matrix_rows + threads_per_block.y - 1) / threads_per_block.y;
  dim3 num_blocks(blocks_x, blocks_y);
  int filter_radius_y = filter_rows / 2;
  int filter_radius_x = filter_cols / 2;
  naive_convolution<<<num_blocks, threads_per_block>>>(d_A, d_B, d_F, matrix_rows, matrix_cols, filter_rows, filter_cols, filter_radius_y, filter_radius_x);
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
  if (verify_convolution(h_A, h_B, h_F, matrix_rows, matrix_cols, filter_rows, filter_cols, filter_radius_y, filter_radius_x) != 0)
  {
    return 1;
  }
  printf("All (sampled) values match\n");

  // Free memory and destroy events
  free(h_A);
  free(h_B);
  free(h_F);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_F));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
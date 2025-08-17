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

constexpr bool random_initialization = false;
constexpr int A_rows = 10000;
constexpr int A_cols = 10000;
constexpr int B_rows = 10000;
constexpr int B_cols = 10000;
constexpr int C_rows = A_rows;
constexpr int C_cols = B_cols;
constexpr int tile_width = 16;
constexpr int coarse_factor = 4;
constexpr int samples_to_check = 10000;

constexpr dim3 threads_per_block(tile_width, tile_width);
constexpr int blocks_x = (C_cols + (threads_per_block.x * coarse_factor) - 1) / (threads_per_block.x * coarse_factor);
constexpr int blocks_y = (C_rows + threads_per_block.y - 1) / threads_per_block.y;
constexpr dim3 num_blocks(blocks_x, blocks_y);

__global__ void tiled_matrix_multiply_coarsened(float *d_A, float *d_B, float *d_C)
{
  int col = blockIdx.x * blockDim.x * coarse_factor + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  int num_tiles = (A_cols + tile_width - 1) / tile_width;
  __shared__ float A_tile[tile_width][tile_width];
  __shared__ float B_tile[tile_width][tile_width];

  float sum[coarse_factor];
  for (int j = 0; j < coarse_factor; j++)
  {
    sum[j] = 0.0f;
  }

  for (int i = 0; i < num_tiles; i++)
  {
    int A_read_row = row;
    int A_read_col = tile_width * i + threadIdx.x;

    if (A_read_row < A_rows && A_read_col < A_cols)
    {
      A_tile[threadIdx.y][threadIdx.x] = d_A[A_read_row * A_cols + A_read_col];
    }
    else
    {
      A_tile[threadIdx.y][threadIdx.x] = 0;
    }

    for (int j = 0; j < coarse_factor; j++)
    {
      int B_read_row = tile_width * i + threadIdx.y;
      int B_read_col = col + j * tile_width;

      if (B_read_row < B_rows && B_read_col < B_cols)
      {
        B_tile[threadIdx.y][threadIdx.x] = d_B[B_read_row * B_cols + B_read_col];
      }
      else
      {
        B_tile[threadIdx.y][threadIdx.x] = 0;
      }

      __syncthreads();

      for (int k = 0; k < tile_width; k++)
      {
        sum[j] += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
      }

      __syncthreads();
    }
  }

  for (int j = 0; j < coarse_factor; j++)
  {
    int write_row = row;
    int write_col = col + j * tile_width;

    if (write_row < A_rows && write_col < B_cols)
    {
      d_C[write_row * B_cols + write_col] = sum[j];
    }
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
  std::vector<float> h_A(A_rows * A_cols);

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
  std::vector<float> h_B(B_rows * B_cols);

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
  std::vector<float> h_C(C_rows * C_cols);

  // Prepare device variables for the matrices
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_C, C_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), B_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Execute the kernel
  tiled_matrix_multiply_coarsened<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, C_memsize, cudaMemcpyDeviceToHost));

  // Check values
  if (verify_matrix_multiplication(h_A.data(), h_B.data(), h_C.data()) != 0)
  {
    return 1;
  }
  printf("All (sampled) values match\n");

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
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

#define MATRIX_ROWS 30000
#define MATRIX_COLS 20000
#define RANDOM_INITIALIZATION false
#define FILTER_RADIUS_X 2
#define FILTER_RADIUS_Y 2
#define FILTER_ROWS (2 * FILTER_RADIUS_Y + 1)
#define FILTER_COLS (2 * FILTER_RADIUS_X + 1)
#define INPUT_TILE_SIZE 16
#define OUTPUT_TILE_SIZE_X (INPUT_TILE_SIZE - 2 * FILTER_RADIUS_X)
#define OUTPUT_TILE_SIZE_Y (INPUT_TILE_SIZE - 2 * FILTER_RADIUS_Y)
#define SAMPLES_TO_CHECK 10000

__constant__ float d_F[2 * FILTER_RADIUS_Y + 1][2 * FILTER_RADIUS_X + 1];

/*
 * Performs a 2D convolution using a tiled memory approach.
 * - Each thread block is responsible for one output tile.
 * - To compute the output tile, the block loads a larger corresponding input tile into shared memory.
 *   The halo region accounts for the filter radius.
 * - The filter coefficients are stored in constant memory for faster access.
 */
__global__ void tiled_convolution(float *d_A, float *d_B, int matrix_rows, int matrix_cols)
{
  signed int col = blockIdx.x * OUTPUT_TILE_SIZE_X + threadIdx.x - FILTER_RADIUS_X;
  signed int row = blockIdx.y * OUTPUT_TILE_SIZE_Y + threadIdx.y - FILTER_RADIUS_Y;

  __shared__ float input_tile[INPUT_TILE_SIZE][INPUT_TILE_SIZE];

  // Load input tile into shared memory using all threads in the block
  if (row >= 0 && row < matrix_rows && col >= 0 && col < matrix_cols)
  {
    input_tile[threadIdx.y][threadIdx.x] = d_A[row * matrix_cols + col];
  }
  else
  {
    input_tile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  // Compute indices of the thread inside the output tile
  signed int output_tile_x = threadIdx.x - FILTER_RADIUS_X;
  signed int output_tile_y = threadIdx.y - FILTER_RADIUS_Y;

  // Check whether the thread should be enabled for the current output tile
  // (threads on the edges are disabled because input tiles are bigger than output tiles)
  if (output_tile_x >= 0 && output_tile_x < OUTPUT_TILE_SIZE_X && output_tile_y >= 0 && output_tile_y < OUTPUT_TILE_SIZE_Y)
  {
    // Check that thread indice s are within bounds,
    // intuitively it is better to have this check nested inside the other one
    // and the profiler agrees (there are ~ 5% less elapsed cycles in the Nsight Compute)
    if (row >= 0 && row < matrix_rows && col >= 0 && col < matrix_cols)
    {
      float sum = 0.0f;

      for (int i = -FILTER_RADIUS_Y; i < FILTER_RADIUS_Y + 1; i++)
      {
        for (int j = -FILTER_RADIUS_X; j < FILTER_RADIUS_X + 1; j++)
        {
          sum += d_F[FILTER_RADIUS_Y + i][FILTER_RADIUS_X + j] * input_tile[threadIdx.y + i][threadIdx.x + j];
        }
      }

      d_B[row * matrix_cols + col] = sum;
    }
  }
}

int verify_convolution(float *h_A, float *h_B, float *h_F, int matrix_rows, int matrix_cols)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<> row_dist(0, matrix_rows - 1);
  std::uniform_int_distribution<> col_dist(0, matrix_cols - 1);

  for (int s = 0; s < SAMPLES_TO_CHECK; s++)
  {
    int i = row_dist(generator);
    int j = col_dist(generator);

    float target_value = h_B[i * matrix_cols + j];
    float expected_value = 0.0f;

    for (int k = -FILTER_RADIUS_Y; k < FILTER_RADIUS_Y + 1; k++)
    {
      for (int l = -FILTER_RADIUS_X; l < FILTER_RADIUS_X + 1; l++)
      {
        if (i + k >= 0 && i + k < matrix_rows && j + l >= 0 && j + l < matrix_cols)
        {
          expected_value += h_A[(i + k) * matrix_cols + (j + l)] * h_F[(FILTER_RADIUS_Y + k) * FILTER_COLS + (FILTER_RADIUS_X + l)];
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
  int matrix_rows = MATRIX_ROWS;
  int matrix_cols = MATRIX_COLS;
  size_t A_memsize = matrix_rows * matrix_cols * sizeof(float);
  float *h_A = (float *)malloc(A_memsize);

  for (int i = 0; i < matrix_rows; i++)
  {
    for (int j = 0; j < matrix_cols; j++)
    {
      if (RANDOM_INITIALIZATION)
      {
        h_A[i * matrix_cols + j] = distribution(generator);
      }
      else
      {
        h_A[i * matrix_cols + j] = 1.0f;
      }
    }
  }

  // Define filter F with values from a random distribution
  size_t F_memsize = FILTER_ROWS * FILTER_COLS * sizeof(float);
  float *h_F = (float *)malloc(F_memsize);

  for (int i = 0; i < FILTER_ROWS; i++)
  {
    for (int j = 0; j < FILTER_COLS; j++)
    {
      if (RANDOM_INITIALIZATION)
        h_F[i * FILTER_COLS + j] = distribution(generator);
      else
      {
        h_F[i * FILTER_COLS + j] = 1.0f;
      }
    }
  }

  // Allocate memory for matrix B which will be the result of the convolution
  // (formally cross-correlation) of the matrix A with the filter F
  // (it will have same size as A, zeros will be used as padding)
  size_t B_memsize = A_memsize;
  float *h_B = (float *)malloc(B_memsize);

  // Prepare device variables for the matrices
  float *d_A, *d_B;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, A_memsize, cudaMemcpyHostToDevice));

  // Copy filter F to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_F, h_F, F_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform convolution on GPU
  dim3 threads_per_block(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
  int blocks_x = (matrix_cols + OUTPUT_TILE_SIZE_X - 1) / OUTPUT_TILE_SIZE_X;
  int blocks_y = (matrix_rows + OUTPUT_TILE_SIZE_Y - 1) / OUTPUT_TILE_SIZE_Y;
  dim3 num_blocks(blocks_x, blocks_y);
  tiled_convolution<<<num_blocks, threads_per_block>>>(d_A, d_B, matrix_rows, matrix_cols);
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
  if (verify_convolution(h_A, h_B, h_F, matrix_rows, matrix_cols) != 0)
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

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
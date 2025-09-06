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
constexpr int matrix_rows = 2048;
constexpr int matrix_cols = 2048;
constexpr int filter_radius_x = 2;
constexpr int filter_radius_y = 2;
constexpr int filter_rows = (2 * filter_radius_y + 1);
constexpr int filter_cols = (2 * filter_radius_x + 1);
constexpr int input_tile_size = 16;
constexpr int output_tile_size_x = (input_tile_size - 2 * filter_radius_x);
constexpr int output_tile_size_y = (input_tile_size - 2 * filter_radius_y);
constexpr int samples_to_check = 10000;

constexpr dim3 threads_per_block(input_tile_size, input_tile_size);
constexpr int blocks_x = (matrix_cols + output_tile_size_x - 1) / output_tile_size_x;
constexpr int blocks_y = (matrix_rows + output_tile_size_y - 1) / output_tile_size_y;
constexpr dim3 num_blocks(blocks_x, blocks_y);

__constant__ float d_F[2 * filter_radius_y + 1][2 * filter_radius_x + 1];

/*
 * Performs a 2D convolution using a tiled memory approach.
 * - Each thread block is responsible for one output tile.
 * - To compute the output tile, the block loads a larger corresponding input tile into shared memory.
 *   The halo region accounts for the filter radius.
 * - The filter coefficients are stored in constant memory for faster access.
 */
__global__ void tiled_convolution(float *d_A, float *d_B)
{
  signed int col = blockIdx.x * output_tile_size_x + threadIdx.x - filter_radius_x;
  signed int row = blockIdx.y * output_tile_size_y + threadIdx.y - filter_radius_y;

  __shared__ float input_tile[input_tile_size][input_tile_size];

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
  signed int output_tile_x = threadIdx.x - filter_radius_x;
  signed int output_tile_y = threadIdx.y - filter_radius_y;

  // Check whether the thread should be enabled for the current output tile
  // (threads on the edges are disabled because input tiles are bigger than output tiles)
  if (output_tile_x >= 0 && output_tile_x < output_tile_size_x && output_tile_y >= 0 && output_tile_y < output_tile_size_y)
  {
    // Check that thread indice s are within bounds,
    // intuitively it is better to have this check nested inside the other one
    // and the profiler agrees (there are ~ 5% less elapsed cycles in the Nsight Compute)
    if (row >= 0 && row < matrix_rows && col >= 0 && col < matrix_cols)
    {
      float sum = 0.0f;

      for (int i = -filter_radius_y; i < filter_radius_y + 1; i++)
      {
        for (int j = -filter_radius_x; j < filter_radius_x + 1; j++)
        {
          sum += d_F[filter_radius_y + i][filter_radius_x + j] * input_tile[threadIdx.y + i][threadIdx.x + j];
        }
      }

      d_B[row * matrix_cols + col] = sum;
    }
  }
}

int verify_convolution(float *h_A, float *h_B, float *h_F)
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
  size_t A_memsize = matrix_rows * matrix_cols * sizeof(float);
  std::vector<float> h_A(matrix_rows * matrix_cols);

  for (int i = 0; i < matrix_rows; i++)
  {
    for (int j = 0; j < matrix_cols; j++)
    {
      if (random_initialization)
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
  size_t F_memsize = filter_rows * filter_cols * sizeof(float);
  std::vector<float> h_F(filter_rows * filter_cols);

  for (int i = 0; i < filter_rows; i++)
  {
    for (int j = 0; j < filter_cols; j++)
    {
      if (random_initialization)
        h_F[i * filter_cols + j] = distribution(generator);
      else
      {
        h_F[i * filter_cols + j] = 1.0f;
      }
    }
  }

  // Allocate memory for matrix B which will be the result of the convolution
  // (formally cross-correlation) of the matrix A with the filter F
  // (it will have same size as A, zeros will be used as padding)
  size_t B_memsize = A_memsize;
  std::vector<float> h_B(matrix_rows * matrix_cols);

  // Prepare device variables for the matrices
  float *d_A, *d_B;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_memsize, cudaMemcpyHostToDevice));

  // Copy filter F to constant memory
  CUDA_CHECK(cudaMemcpyToSymbol(d_F, h_F.data(), F_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform convolution on GPU
  tiled_convolution<<<num_blocks, threads_per_block>>>(d_A, d_B);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms (use NCU for a more precise measurement!)\n", milliseconds);

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, B_memsize, cudaMemcpyDeviceToHost));

  // Check values
  if (verify_convolution(h_A.data(), h_B.data(), h_F.data()) != 0)
  {
    return 1;
  }
  printf("All (sampled) values match\n");

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
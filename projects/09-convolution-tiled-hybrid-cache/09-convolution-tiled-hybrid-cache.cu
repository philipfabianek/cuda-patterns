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
constexpr int tile_size = 16;
constexpr int samples_to_check = 10000;

constexpr dim3 threads_per_block(tile_size, tile_size);
constexpr int blocks_x = (matrix_cols + threads_per_block.x - 1) / threads_per_block.x;
constexpr int blocks_y = (matrix_rows + threads_per_block.y - 1) / threads_per_block.y;
constexpr dim3 num_blocks(blocks_x, blocks_y);

__constant__ float d_F[2 * filter_radius_y + 1][2 * filter_radius_x + 1];

/*
 * Performs a 2D convolution using a hybrid cache approach.
 * - Each thread block is responsible for one output tile.
 * - To compute the output tile, the block loads a corresponding input tile into shared memory.
 *   The input tile has the same size as the output tile, so the elements
 *   outside which are needed for the convolution are loaded from global memory.
 *   This is an optimistic approach which hopes for good cache hits
 *   but according to my tests it performs significantly worse than the previous tiled approach
 *   from 08-tiled-convolution.
 */
__global__ void tiled_convolution(float *d_A, float *d_B)
{
  signed int col = blockIdx.x * blockDim.x + threadIdx.x;
  signed int row = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ float input_tile[tile_size][tile_size];

  // Load input tile into shared memory using all threads in the block
  if (row < matrix_rows && col < matrix_cols)
  {
    input_tile[threadIdx.y][threadIdx.x] = d_A[row * matrix_cols + col];
  }
  else
  {
    input_tile[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  // Check that thread indices are within bounds
  if (row < matrix_rows && col < matrix_cols)
  {
    float sum = 0.0f;

    for (int i = -filter_radius_y; i < filter_radius_y + 1; i++)
    {
      for (int j = -filter_radius_x; j < filter_radius_x + 1; j++)
      {
        int input_x = threadIdx.x + j;
        int input_y = threadIdx.y + i;

        if (input_x >= 0 && input_x < tile_size && input_y >= 0 && input_y < tile_size)
        {
          // Use shared memory
          sum += d_F[filter_radius_y + i][filter_radius_x + j] * input_tile[input_y][input_x];
        }
        else
        {
          // Use global memory, a lot of these accesses will be cached,
          // here we need to check bounds again
          if (row + i >= 0 && row + i < matrix_rows && col + j >= 0 && col + j < matrix_cols)
          {
            sum += d_F[filter_radius_y + i][filter_radius_x + j] * d_A[(row + i) * matrix_cols + (col + j)];
          }
        }
      }
    }

    d_B[row * matrix_cols + col] = sum;
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
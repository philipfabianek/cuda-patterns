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

constexpr int matrix_rows = 50000;
constexpr int matrix_cols = 40000;
constexpr float sparsity = 0.01f; // 1% of elements are non-zero
constexpr int num_non_zeros = matrix_rows * matrix_cols * sparsity;

constexpr int threads_per_block = 256;
constexpr int blocks_per_grid = (num_non_zeros + threads_per_block - 1) / threads_per_block;

/*
 * Performs Sparse Matrix-Vector multiplication (SpMV) using the coordinate (COO) format.
 * - Each thread computes the product of one non-zero element of the matrix
 * - atomicAdd is used to prevent race conditions
 */
__global__ void spmv_coo_kernel(const float *value, const int *rowIdx, const int *colIdx,
                                const float *x, float *y)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_non_zeros)
  {
    float v = value[idx];
    int r = rowIdx[idx];
    int c = colIdx[idx];
    atomicAdd(&y[r], v * x[c]);
  }
}

/*
 * Verifies the result of the SpMV computation from the GPU
 * by comparing it against a CPU-based computation.
 */
int verify_spmv(const float *h_value, const int *h_rowIdx, const int *h_colIdx, const float *h_x,
                const float *h_y_gpu, int n_non_zeros, int n_rows)
{
  float *h_y_cpu = (float *)calloc(n_rows, sizeof(float));

  for (int i = 0; i < n_non_zeros; ++i)
  {
    h_y_cpu[h_rowIdx[i]] += h_value[i] * h_x[h_colIdx[i]];
  }

  int status = 0;
  for (int i = 0; i < n_rows; ++i)
  {
    if (fabs(h_y_cpu[i] - h_y_gpu[i]) > 1e-3)
    {
      printf("Mismatch at row %d: expected %f, got %f\n", i, h_y_cpu[i], h_y_gpu[i]);
      status = 1;
      break;
    }
  }

  free(h_y_cpu);
  return status;
}

int main()
{
  // Create random number generator and random distribution
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> value_dist(-1.0f, 1.0f);
  std::uniform_int_distribution<int> row_dist(0, matrix_rows - 1);
  std::uniform_int_distribution<int> col_dist(0, matrix_cols - 1);

  // Define host sparse matrix in COO format with values from the random distribution
  float *h_value = (float *)malloc(num_non_zeros * sizeof(float));
  int *h_rowIdx = (int *)malloc(num_non_zeros * sizeof(int));
  int *h_colIdx = (int *)malloc(num_non_zeros * sizeof(int));

  for (int i = 0; i < num_non_zeros; ++i)
  {
    h_value[i] = value_dist(generator);
    h_rowIdx[i] = row_dist(generator);
    h_colIdx[i] = col_dist(generator);
  }

  // Define host input vector
  float *h_x = (float *)malloc(matrix_cols * sizeof(float));
  for (int i = 0; i < matrix_cols; ++i)
  {
    h_x[i] = value_dist(generator);
  }

  // Allocate memory for host output vector
  float *h_y = (float *)malloc(matrix_rows * sizeof(float));

  // Prepare device variables
  size_t value_memsize = num_non_zeros * sizeof(float);
  size_t idx_memsize = num_non_zeros * sizeof(int);
  size_t x_memsize = matrix_cols * sizeof(float);
  size_t y_memsize = matrix_rows * sizeof(float);

  float *d_value, *d_x, *d_y;
  int *d_rowIdx, *d_colIdx;

  CUDA_CHECK(cudaMalloc((void **)&d_value, value_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_rowIdx, idx_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_colIdx, idx_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_x, x_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_y, y_memsize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_value, h_value, value_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rowIdx, h_rowIdx, idx_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx, idx_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, x_memsize, cudaMemcpyHostToDevice));

  // Ensure output vector is initialized to zero
  CUDA_CHECK(cudaMemset(d_y, 0, y_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Compute sparse matrix-vector multiplication on GPU
  spmv_coo_kernel<<<blocks_per_grid, threads_per_block>>>(d_value, d_rowIdx, d_colIdx, d_x, d_y);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Copy data from device to host
  CUDA_CHECK(cudaMemcpy(h_y, d_y, y_memsize, cudaMemcpyDeviceToHost));

  // Verify the result
  if (verify_spmv(h_value, h_rowIdx, h_colIdx, h_x, h_y, num_non_zeros, matrix_rows) != 0)
  {
    printf("Verification failed\n");
  }
  else
  {
    printf("All values match\n");
  }

  // Free memory and destroy events
  free(h_value);
  free(h_rowIdx);
  free(h_colIdx);
  free(h_x);
  free(h_y);

  CUDA_CHECK(cudaFree(d_value));
  CUDA_CHECK(cudaFree(d_rowIdx));
  CUDA_CHECK(cudaFree(d_colIdx));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

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
// One thread per row
constexpr int blocks_per_grid = (matrix_rows + threads_per_block - 1) / threads_per_block;

/*
 * Performs Sparse Matrix-Vector multiplication (SpMV) using the ELLPACK (ELL) format.
 * - Each thread is responsible for one row of the matrix.
 * - Since each thread writes to its own unique row,
 *   no atomic operations are needed.
 */
__global__ void spmv_ell_kernel(const float *ell_data, const int *ell_col_indices,
                                const float *x, float *y, int max_nnz_per_row)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < matrix_rows)
  {
    float sum = 0.0f;
    for (int i = 0; i < max_nnz_per_row; ++i)
    {
      int idx = i * matrix_rows + row;
      float value = ell_data[idx];
      int col = ell_col_indices[idx];

      if (value != 0.0f)
      {
        sum += value * x[col];
      }
    }
    y[row] = sum;
  }
}

/*
 * Verifies the result of the SpMV computation from the GPU
 * by comparing it against a CPU-based computation (using the original COO data).
 */
int verify_spmv(const float *h_value_coo, const int *h_rowIdx_coo,
                const int *h_colIdx_coo, const float *h_x, const float *h_y_gpu)
{
  float *h_y_cpu = (float *)calloc(matrix_rows, sizeof(float));

  for (int i = 0; i < num_non_zeros; ++i)
  {
    h_y_cpu[h_rowIdx_coo[i]] += h_value_coo[i] * h_x[h_colIdx_coo[i]];
  }

  int status = 0;
  for (int i = 0; i < matrix_rows; ++i)
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
  std::vector<float> h_value_coo(num_non_zeros);
  std::vector<int> h_rowIdx_coo(num_non_zeros);
  std::vector<int> h_colIdx_coo(num_non_zeros);

  for (int i = 0; i < num_non_zeros; ++i)
  {
    h_value_coo[i] = value_dist(generator);
    h_rowIdx_coo[i] = row_dist(generator);
    h_colIdx_coo[i] = col_dist(generator);
  }

  // Convert COO to ELL, first compute the maximum number of non-zero elements
  // per row across all rows which tells us the size of the padded matrix
  std::vector<int> nnz_per_row(matrix_rows, 0);
  for (int r : h_rowIdx_coo)
  {
    nnz_per_row[r]++;
  }
  int max_nnz_per_row = *std::max_element(nnz_per_row.begin(), nnz_per_row.end());

  // Construct vectors describing the matrix in the ELL format,
  // it is necessary to use an auxiliary variable row_nnz_counts
  // which holds the number of non-zero elements for each row
  // that were already assigned to compute the offsets
  size_t num_elements = (size_t)matrix_rows * max_nnz_per_row;
  std::vector<float> h_value_ell(num_elements, 0.0f);
  std::vector<int> h_colIdx_ell(num_elements, 0);
  std::vector<int> row_nnz_counts(matrix_rows, 0);

  for (int i = 0; i < num_non_zeros; ++i)
  {
    int r = h_rowIdx_coo[i];
    int c = h_colIdx_coo[i];
    float v = h_value_coo[i];

    int &nnz_count = row_nnz_counts[r];
    int idx = nnz_count * matrix_rows + r;

    h_value_ell[idx] = v;
    h_colIdx_ell[idx] = c;
    nnz_count++;
  }

  // Define host input vector
  std::vector<float> h_x(matrix_cols);
  for (int i = 0; i < matrix_cols; ++i)
  {
    h_x[i] = value_dist(generator);
  }

  // Allocate memory for host output vector
  std::vector<float> h_y(matrix_rows);

  // Prepare device variables
  size_t value_memsize = num_elements * sizeof(float);
  size_t idx_memsize = num_elements * sizeof(int);
  size_t x_memsize = matrix_cols * sizeof(float);
  size_t y_memsize = matrix_rows * sizeof(float);

  float *d_value, *d_x, *d_y;
  int *d_colIdx;

  CUDA_CHECK(cudaMalloc((void **)&d_value, value_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_colIdx, idx_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_x, x_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_y, y_memsize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_value, h_value_ell.data(), value_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_colIdx, h_colIdx_ell.data(), idx_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), x_memsize, cudaMemcpyHostToDevice));

  // Ensure output vector is initialized to zero
  CUDA_CHECK(cudaMemset(d_y, 0, y_memsize));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Compute sparse matrix-vector multiplication on GPU
  spmv_ell_kernel<<<blocks_per_grid, threads_per_block>>>(d_value, d_colIdx, d_x, d_y, max_nnz_per_row);
  CUDA_CHECK(cudaGetLastError());

  // Record the end time and synchronize
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Calculate elapsed time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Copy data from device to host
  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, y_memsize, cudaMemcpyDeviceToHost));

  // Verify the result (using the original COO data)
  if (verify_spmv(h_value_coo.data(), h_rowIdx_coo.data(), h_colIdx_coo.data(), h_x.data(), h_y.data()) != 0)
  {
    printf("Verification failed\n");
  }
  else
  {
    printf("All values match\n");
  }

  // Free memory and destroy events
  CUDA_CHECK(cudaFree(d_value));
  CUDA_CHECK(cudaFree(d_colIdx));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
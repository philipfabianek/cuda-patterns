#include <stdio.h>
#include <vector>

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

constexpr int cols = 256;
constexpr int rows = 256;
constexpr size_t mem_size = cols * rows * sizeof(int);

constexpr dim3 threads_per_block(16, 16);
constexpr int blocks_x = (cols + threads_per_block.x - 1) / threads_per_block.x;
constexpr int blocks_y = (rows + threads_per_block.y - 1) / threads_per_block.y;
constexpr dim3 num_blocks(blocks_x, blocks_y);

__global__ void matrix_add(int *d_A, int *d_B, int *d_C, int _rows, int _cols)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < _rows && col < _cols)
  {
    d_C[row * _cols + col] = d_A[row * _cols + col] + d_B[row * _cols + col];
  }
}

int main()
{
  // Define host matrices
  std::vector<int> h_A(cols * rows);
  std::vector<int> h_B(cols * rows);
  std::vector<int> h_C(cols * rows);

  for (int row = 0; row < rows; row++)
  {
    for (int col = 0; col < cols; col++)
    {
      h_A[row * cols + col] = row;
      h_B[row * cols + col] = row + col;
    }
  }

  // Prepare device variables
  int *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_B, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_C, mem_size));

  // Print top-left submatrices
  printf("Top-left 5x5 submatrix of A\n");

  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      printf("%d ", h_A[row * cols + col]);
    }
    printf("\n");
  }

  printf("\n");
  printf("Top-left 5x5 submatrix of B\n");

  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      printf("%d ", h_B[row * cols + col]);
    }
    printf("\n");
  }

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), mem_size, cudaMemcpyHostToDevice));

  // Perform matrix addition on GPU
  matrix_add<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, rows, cols);
  CUDA_CHECK(cudaGetLastError());

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, mem_size, cudaMemcpyDeviceToHost));

  // Print top-left result submatrix
  printf("\n");
  printf("Top-left 5x5 submatrix of C\n");

  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      printf("%d ", h_C[row * cols + col]);
    }
    printf("\n");
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

__global__ void naive_matrix_multiply(int *d_A, int *d_B, int *d_C, int A_rows, int A_cols, int B_rows, int B_cols)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < A_rows && col < B_cols)
  {
    int sum = 0;
    for (int k = 0; k < A_cols; k++)
    {
      sum += d_A[row * A_cols + k] * d_B[k * B_cols + col];
    }
    d_C[row * B_cols + col] = sum;
  }
}

int main()
{
  // Define matrix A
  int A_rows = 8;
  int A_cols = 10;
  size_t A_memsize = A_rows * A_cols * sizeof(int);
  int *h_A = (int *)malloc(A_memsize);

  for (int i = 0; i < A_rows; i++)
  {
    for (int j = 0; j < A_cols; j++)
    {
      h_A[i * A_cols + j] = i * A_cols + j;
    }
  }

  // Print row 1 of A
  printf("Row 1 of A\n");
  for (int j = 0; j < A_cols; j++)
  {
    printf("%d ", h_A[A_cols + j]);
  }
  printf("\n");

  // Define matrix B
  int B_rows = 10;
  int B_cols = 12;
  size_t B_memsize = B_rows * B_cols * sizeof(int);
  int *h_B = (int *)malloc(B_memsize);

  for (int i = 0; i < B_rows; i++)
  {
    for (int j = 0; j < B_cols; j++)
    {
      h_B[i * B_cols + j] = i * B_cols + j;
    }
  }

  // Print column 2 of B
  printf("Column 2 of B\n");
  for (int i = 0; i < B_rows; i++)
  {
    printf("%d ", h_B[i * B_cols + 2]);
  }
  printf("\n");

  // Allocate memory for matrix C which will be the product of A and B
  int C_rows = A_rows;
  int C_cols = B_cols;
  size_t C_memsize = C_rows * C_cols * sizeof(int);
  int *h_C = (int *)malloc(C_memsize);

  // Prepare device variables for the matrices
  int *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_C, C_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, A_memsize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, B_memsize, cudaMemcpyHostToDevice));

  // Perform matrix multiplication on GPU
  dim3 threads_per_block(16, 16);
  int blocks_x = (C_cols + threads_per_block.x - 1) / threads_per_block.x;
  int blocks_y = (C_rows + threads_per_block.y - 1) / threads_per_block.y;
  dim3 num_blocks(blocks_x, blocks_y);
  naive_matrix_multiply<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, A_rows, A_cols, B_rows, B_cols);
  CUDA_CHECK(cudaGetLastError());

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_C, d_C, C_memsize, cudaMemcpyDeviceToHost));

  // Check C[1, 2]
  printf("C[1, 2] (should be 9110)\n");
  printf("%d\n", h_C[C_cols + 2]);

  // Free memory
  free(h_A);
  free(h_B);
  free(h_C);

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}
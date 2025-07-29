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

constexpr int N = 1000;
constexpr size_t mem_size = N * sizeof(int);
constexpr int threads_per_block = 256;
constexpr int blocks = (N + threads_per_block - 1) / threads_per_block;

__global__ void vector_add(int *d_a, int *d_b, int *d_c)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

int main()
{
  int *h_a = (int *)malloc(mem_size);
  int *h_b = (int *)malloc(mem_size);
  int *h_c = (int *)malloc(mem_size);

  if (h_a == NULL || h_b == NULL || h_c == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors\n");
    return 1;
  }

  int *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size));

  for (int i = 0; i < N; i++)
  {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  CUDA_CHECK(cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice));

  vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; i++)
  {
    printf("%d\n", h_c[i]);
  }

  free(h_a);
  free(h_b);
  free(h_c);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
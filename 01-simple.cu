#include <stdio.h>
#include <stdlib.h>

const int N = 1000;
const size_t mem_size = N * sizeof(int);
const int threads_per_block = 256;
const int blocks = (N + threads_per_block - 1) / threads_per_block;

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

  int *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, mem_size);
  cudaMalloc((void **)&d_b, mem_size);
  cudaMalloc((void **)&d_c, mem_size);

  for (int i = 0; i < N; i++)
  {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, mem_size, cudaMemcpyHostToDevice);

  vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c);

  cudaMemcpy(h_c, d_c, mem_size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++)
  {
    printf("%d\n", h_c[i]);
  }

  free(h_a);
  free(h_b);
  free(h_c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
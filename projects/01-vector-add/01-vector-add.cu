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
  // Define input arrays
  std::vector<int> h_a(N);
  std::vector<int> h_b(N);
  std::vector<int> h_c(N);

  for (int i = 0; i < N; i++)
  {
    h_a[i] = i;
    h_b[i] = i * i;
  }

  // Prepare device variables
  int *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc((void **)&d_a, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_b, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_c, mem_size));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), mem_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), mem_size, cudaMemcpyHostToDevice));

  // Perform vector addition on GPU
  vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c);
  CUDA_CHECK(cudaGetLastError());

  // Move data from device to host
  CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, mem_size, cudaMemcpyDeviceToHost));

  // Check first 10 elements of the result
  for (int i = 0; i < 10; i++)
  {
    printf("%d\n", h_c[i]);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}
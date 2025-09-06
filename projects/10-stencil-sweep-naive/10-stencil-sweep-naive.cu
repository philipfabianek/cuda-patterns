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
constexpr int N = 512;
constexpr int block_size = 8;
constexpr int samples_to_check = 100000;

constexpr dim3 threads_per_block(block_size, block_size, block_size);
constexpr int blocks_x = (N + block_size - 1) / block_size;
constexpr int blocks_y = (N + block_size - 1) / block_size;
constexpr int blocks_z = (N + block_size - 1) / block_size;
constexpr dim3 num_blocks(blocks_x, blocks_y, blocks_z);

/*
 * Performs a 3D stencil sweep on a 3D tensor using a 7-point stencil.
 * - Each thread computes one element of the output tensor.
 * - Boundary elements are set to zero.
 */
__global__ void stencil_kernel(float *d_A, float *d_B)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= 1 &&
      x < N - 1 &&
      y >= 1 &&
      y < N - 1 &&
      z >= 1 &&
      z < N - 1)
  {
    // Approximate inner values using a 7-point stencil
    float sum = d_A[z * N * N + y * N + x] +
                d_A[(z - 1) * N * N + y * N + x] +
                d_A[(z + 1) * N * N + y * N + x] +
                d_A[z * N * N + (y - 1) * N + x] +
                d_A[z * N * N + (y + 1) * N + x] +
                d_A[z * N * N + y * N + (x - 1)] +
                d_A[z * N * N + y * N + (x + 1)];
    d_B[z * N * N + y * N + x] = sum;
  }
  else
  {
    // Set boundary values to zero
    if (x < N && y < N && z < N)
    {
      d_B[z * N * N + y * N + x] = 0.0f;
    }
  }
}

int verify_stencil_sweep(float *h_A, float *h_B)
{
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<> x_dist(0, N - 1);
  std::uniform_int_distribution<> y_dist(0, N - 1);
  std::uniform_int_distribution<> z_dist(0, N - 1);

  for (int s = 0; s < samples_to_check; s++)
  {
    int i = x_dist(generator);
    int j = y_dist(generator);
    int k = z_dist(generator);

    float target_value = h_B[k * N * N + j * N + i];
    float expected_value = 0.0f;

    if (i >= 1 && i < N - 1 &&
        j >= 1 && j < N - 1 &&
        k >= 1 && k < N - 1)
    {
      expected_value = h_A[k * N * N + j * N + i] +
                       h_A[(k - 1) * N * N + j * N + i] +
                       h_A[(k + 1) * N * N + j * N + i] +
                       h_A[k * N * N + (j - 1) * N + i] +
                       h_A[k * N * N + (j + 1) * N + i] +
                       h_A[k * N * N + j * N + (i - 1)] +
                       h_A[k * N * N + j * N + (i + 1)];
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

  // Define 3D data tensor
  size_t A_memsize = N * N * N * sizeof(float);
  std::vector<float> h_A(N * N * N);

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      for (int k = 0; k < N; k++)
      {
        if (random_initialization)
        {
          h_A[i * N * N + j * N + k] = distribution(generator);
        }
        else
        {
          h_A[i * N * N + j * N + k] = 1.0f;
        }
      }
    }
  }

  // Allocate memory for tensor B which will be the result
  // of applying the stencil kernel to the tensor A
  size_t B_memsize = A_memsize;
  std::vector<float> h_B(N * N * N);

  // Prepare device variables for the tensors
  float *d_A, *d_B;
  CUDA_CHECK(cudaMalloc((void **)&d_A, A_memsize));
  CUDA_CHECK(cudaMalloc((void **)&d_B, B_memsize));

  // Move data from host to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), A_memsize, cudaMemcpyHostToDevice));

  // Create events for timing
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Perform stencil sweep on GPU
  stencil_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B);
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
  if (verify_stencil_sweep(h_A.data(), h_B.data()) != 0)
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
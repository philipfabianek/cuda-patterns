#include <stdio.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

constexpr int blur_radius = 135;
constexpr dim3 threads_per_block(16, 16, 1);

__global__ void blur_kernel(uchar3 *d_input_image, uchar3 *d_output_image, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    float3 sum = {0.0f, 0.0f, 0.0f};
    int count = 0;

    for (int dx = -blur_radius; dx <= blur_radius; dx++)
    {
      for (int dy = -blur_radius; dy <= blur_radius; dy++)
      {

        int neighbor_col = col + dx;
        int neighbor_row = row + dy;

        if (-1 < neighbor_col && neighbor_col < width && -1 < neighbor_row && neighbor_row < height)
        {
          count += 1;

          int neighbor_idx = neighbor_row * width + neighbor_col;
          uchar3 neighbor_pixel = d_input_image[neighbor_idx];

          sum.x += neighbor_pixel.x;
          sum.y += neighbor_pixel.y;
          sum.z += neighbor_pixel.z;
        }
      }
    }

    int output_idx = row * width + col;
    d_output_image[output_idx] = make_uchar3(sum.x / count, sum.y / count, sum.z / count);
  }
}

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
    return 1;
  }

  const char *input_filename = argv[1];
  const char *output_filename = argv[2];

  printf("Will read from %s and write to %s\n", input_filename, output_filename);

  // Load input image from file
  int width, height, channels;
  unsigned char *h_input_image_stb = stbi_load(input_filename, &width, &height, &channels, 3);
  if (h_input_image_stb == NULL)
  {
    fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
    return 1;
  }

  printf("Loaded image %s (%d x %d)\n", input_filename, width, height);

  size_t mem_size = (size_t)width * height * 3 * sizeof(unsigned char);

  // Define host input and output image vectors
  std::vector<unsigned char> h_input_image(h_input_image_stb, h_input_image_stb + mem_size);
  std::vector<unsigned char> h_output_image(mem_size);

  stbi_image_free(h_input_image_stb);

  // Prepare device variables for input and output images
  unsigned char *d_input_image;
  unsigned char *d_output_image;
  CUDA_CHECK(cudaMalloc((void **)&d_input_image, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_output_image, mem_size));

  // Move input image data from host to device
  CUDA_CHECK(cudaMemcpy(d_input_image, h_input_image.data(), mem_size, cudaMemcpyHostToDevice));

  // Perform image blur on GPU
  int blocks_x = (width + threads_per_block.x - 1) / threads_per_block.x;
  int blocks_y = (height + threads_per_block.y - 1) / threads_per_block.y;
  dim3 num_blocks(blocks_x, blocks_y);
  blur_kernel<<<num_blocks, threads_per_block>>>((uchar3 *)d_input_image, (uchar3 *)d_output_image, width, height);
  CUDA_CHECK(cudaGetLastError());

  // Move output image data from device to host
  CUDA_CHECK(cudaMemcpy(h_output_image.data(), d_output_image, mem_size, cudaMemcpyDeviceToHost));

  // Write output image to file
  int success = stbi_write_png(output_filename, width, height, 3, h_output_image.data(), width * 3);
  if (!success)
  {
    fprintf(stderr, "Error writing image %s\n", output_filename);
  }
  else
  {
    printf("Successfully wrote image to %s\n", output_filename);
  }

  // Free device memory
  CUDA_CHECK(cudaFree(d_input_image));
  CUDA_CHECK(cudaFree(d_output_image));

  return 0;
}
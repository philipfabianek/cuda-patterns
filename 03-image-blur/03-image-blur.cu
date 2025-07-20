#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#define CUDA_CHECK(err)                                                                          \
  {                                                                                              \
    if (err != cudaSuccess)                                                                      \
    {                                                                                            \
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

const int blur_radius = 7;

__global__ void blur_kernel(unsigned char *d_input_image, unsigned char *d_output_image, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height)
  {
    int r_sum = 0;
    int g_sum = 0;
    int b_sum = 0;
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
          int idx = (neighbor_row * width + neighbor_col) * 3;
          r_sum += d_input_image[idx];
          g_sum += d_input_image[idx + 1];
          b_sum += d_input_image[idx + 2];
        }
      }
    }

    int idx = (row * width + col) * 3;

    d_output_image[idx] = r_sum / count;
    d_output_image[idx + 1] = g_sum / count;
    d_output_image[idx + 2] = b_sum / count;
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

  int width, height, channels;
  unsigned char *h_input_image = stbi_load(input_filename, &width, &height, &channels, 3);
  if (h_input_image == NULL)
  {
    fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
    return 1;
  }

  printf("Loaded image %s (%d x %d)\n", input_filename, width, height);

  size_t mem_size = width * height * 3 * sizeof(unsigned char);

  unsigned char *h_output_image = (unsigned char *)malloc(mem_size);
  if (h_output_image == NULL)
  {
    fprintf(stderr, "Failed to allocate memory for output image\n");
    stbi_image_free(h_input_image);
    return 1;
  }

  unsigned char *d_input_image;
  unsigned char *d_output_image;
  CUDA_CHECK(cudaMalloc((void **)&d_input_image, mem_size));
  CUDA_CHECK(cudaMalloc((void **)&d_output_image, mem_size));

  CUDA_CHECK(cudaMemcpy(d_input_image, h_input_image, mem_size, cudaMemcpyHostToDevice));

  dim3 threads_per_block(16, 16, 1);
  const int blocks_x = (width + threads_per_block.x - 1) / threads_per_block.x;
  const int blocks_y = (height + threads_per_block.y - 1) / threads_per_block.y;
  const dim3 num_blocks(blocks_x, blocks_y);
  blur_kernel<<<num_blocks, threads_per_block>>>(d_input_image, d_output_image, width, height);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_output_image, d_output_image, mem_size, cudaMemcpyDeviceToHost));

  int success = stbi_write_png(output_filename, width, height, 3, h_output_image, width * 3);
  if (!success)
  {
    fprintf(stderr, "Error writing image %s\n", output_filename);
  }
  else
  {
    printf("Successfully wrote image to %s\n", output_filename);
  }

  stbi_image_free(h_input_image);
  free(h_output_image);
  CUDA_CHECK(cudaFree(d_input_image));
  CUDA_CHECK(cudaFree(d_output_image));

  return 0;
}
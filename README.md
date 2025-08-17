# CUDA patterns

This repository is a collection of CUDA projects that implement and optimize the fundamental algorithms, patterns and techniques in GPU programming. The projects are based on concepts from the famous "Programming Massively Parallel Processors" (PMPP) textbook.

Each implementation was profiled with NVIDIA Nsight Compute to inspect metrics like kernel duration, occupancies and various throughputs. This approach led to various kernel optimizations and hyperparameter tuning (for my GPU) in the more complex projects. All reported speedups are for kernel execution time only and do not include host-device memory transfers.

Noteworthy projects include:

- **`17-parallel-histogram-hybrid`**: An optimized histogram kernel using shared memory, thread coarsening and sequential memory access via a grid-stride loop. Hyperparameters were finetuned with an automated grid search. The result is a ~1,000x speedup over single-threaded CPU implementation.
- **`21-prefix-sum-general`**: A scalable, multi-block parallel prefix sum (inclusive scan) implementation. It uses a hierarchical two phase approach (up-sweep and down-sweep) to build and traverse a reduction tree. The result is a ~30x speedup over a single-threaded CPU implementation.

## Building the Projects

This repository uses CMake to build all 23 projects.

### 1. Prerequisites

- A C++ compiler (g++, clang++, etc.)
- The NVIDIA CUDA Toolkit (nvcc)
- CMake (version 3.18 or higher)

### 2. Build Steps

From the root of the repository, run the following commands:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will compile all projects. The executables will be located in the `build/` directory.

## Project Overview

### [`01-vector-add`](./01-vector-add/)

In this project, I familiarized myself with the standard CUDA program skeleton that consists of device memory allocation, host-to-device data transfer, executing the kernel and device-to-host data transfer. I also learned about the hierarchy of **grids**, **blocks** and **threads**.

---

### [`02-matrix-add`](./02-matrix-add)

In this project, I implemented basic matrix addition kernel. I learned about using 2D grids and the 2D-to-1D coordinate mapping using **strides**.

---

### [`03-image-blur`](./03-image-blur)

In this project, I implemented basic box blur kernel for images, where each thread computes one output pixel by averaging its neighbors.

This approach can lead to a very common memory bottleneck from redundant global memory reads. However, that is not the case on my GPU, since the SM (compute) throughput is over 80% for various blur box sizes. This is most likely due to an effective utilization of the L1 cache (also over 80% throughput).

This is also one of the more practical projects as it uses the `stb` library to load and save image files.

---

### [`04-matrix-multiply-naive`](./04-matrix-multiply-naive)

In this project, I implemented a naive matrix multiplication kernel, where each thread computes one output element by reading a full row from the first matrix and a full column from the second matrix.

Profiling revealed a similar situation as in the previous project: Traditionally **memory-bound** problem is actually **compute-bound** on my GPU with SM (compute) throughput over 80% for large matrices. This is again most likely due to highly effective caching.

---

### [`05-matrix-multiply-tiled`](./05-matrix-multiply-tiled)

In this project, I implemented the classical tiled matrix multiplication kernel. Threads in a block cooperate to load tiles from the input matrices into the **shared memory**. This reduces slow global memory access and can help with memory-bound problems.

Since the naive matrix multiplication kernel was already compute-bound, this lead only to minor performance improvements on my GPU. It was still faster though because of the low latency of shared memory reads.

---

### [`06-matrix-multiply-tiled-coarsened`](./06-matrix-multiply-tiled-coarsened)

In this project, I extended the tiled matrix multiplication kernel with **thread coarsening**. Each thread is responsible for several elements (determined by a coarse factor).

This can improve compute efficiency by doing more work per thread and reducing the "price of parallelism", e.g. **loop or synchronization overhead**.

While coarsening provided a minor performance boost with a specific coarse factor, with other factor values it lead to a significant performance decrease.

---

### [`07-convolution-naive`](./07-convolution-naive)

In this project, I implemented a naive 2D convolution kernel. This is a generalization of the image blur kernel, where each thread computes one output element by performing a dot product between the input window and the filter kernel.

Following the pattern from the previous projects, this traditionally memory-bound algorithm is once again compute-bound on my GPU with SM (compute) throughput over 80% for large matrices. This is again most likely due to highly effective caching.

### [`08-convolution-tiled`](./08-convolution-tiled)

In this project, I implemented a tiled 2D convolution kernel. Threads in a block cooperate to load a larger input tile into the shared memory. The larger input tile contains a **halo region** which is required to compute the convolutions for pixels near the edge of the output tile.

Since the naive convolution kernel was already compute-bound, this lead only to minor performance improvements on my GPU. It was still faster though because of the low latency of shared memory reads.

---

### [`09-convolution-tiled-hybrid-cache`](./09-convolution-tiled-hybrid-cache)

In this project, I experimented with a **hybrid memory access pattern** for the tiled convolution. Threads still cooperate to load the core of the input tile into the shared memory but now access the values from the halo region from global memory.

This is an optimistic approach which hopes for good cache hits but it performs significantly worse on my GPU than both of the previous approaches. This is most likely because the global memory reads are scattered.

---

### [`10-stencil-sweep-naive`](./10-stencil-sweep-naive)

In this project, I implemented a naive 3D 7-point stencil sweep kernel. While this is a specific instance of convolution the optimization opportunities are slightly different.

Profiling revealed over 90% memory throughput with SM (compute) throughput of over 40%, resulting in a memory-bound (latency-bound) problem.

---

### [`11-stencil-sweep-tiled`](./11-stencil-sweep-tiled)

In this project, I implemented a tiled 3D 7-point stencil sweep kernel. Threads in a block cooperate to load a larger input tile into the shared memory, including a 1-element halo.

This approach lead resulted in balanced throughputs, however it lead to significant performance decrease overall, most likely because of the additional instruction overhead (thread synchronization, boundary checks) from the tiling logic.

---

### [`12-stencil-sweep-tiled-plane-sweep`](./12-stencil-sweep-tiled-plane-sweep)

In this project, I implemented a 3D 7-point stencil sweep kernel using a **plane sweep** technique.

This algorithm iterates along the z-axis. Since the stencil radius is 1, to compute an output plane, only the previous, current and next input planes are required.
By storing the current xy-plane in the shared memory and keeping the previous and next z-values in registers this approach drastically reduces shared memory usage compared to the previous full 3D tile.

The reduction in shared memory usage allowed larger blocks to be used and the result was a slight performance increase over the previously fastest naive stencil implementation.

---

### [`13-parallel-histogram-naive`](./13-parallel-histogram-naive)

In this project, I implemented a naive parallel histogram kernel, which computes character frequencies.
Results are written to global memory, with race conditions prevented by using `atomicAdd`.

This creates a massive bottleneck since threads are forced to wait for the lock release. The result is extremely poor performance with single-digit SM and memory throughputs.

---

### [`14-parallel-histogram-privatized`](./14-parallel-histogram-privatized)

In this project, I extended the parallel histogram kernel by using **privatization**.

Histograms for each block are stored in a separate, conceptually private, section of the global memory. These partial results are then merged into the final, global histogram.

This alleviates the contention of the atomic operations and significantly improves the performance, resulting in a speedup of ~10x on my GPU over the naive version.

---

### [`15-parallel-histogram-shared`](./15-parallel-histogram-shared)

In this project, I extended the previous privatized parallel histogram kernel by using shared memory.

Each block computes its partial histogram using shared memory.
This is a very natural optimization which in this case significantly reduces the number of global writes.

The result is a ~4x speedup over the previous version.

---

### [`16-parallel-histogram-shared-coarsened`](./16-parallel-histogram-shared-coarsened)

In this project, I extended the shared memory parallel histogram kernel by using thread coarsening. Each thread processes a number of elements depending on the coarse factor.

This reduces the number of global `atomicAdd` operations and also reduces the loop synchronization overhead. In this implementation, the memory access pattern of threads in a warp is **interleaved** resulting in perfect memory coalescing.

This approach leads to a further ~2.5x speedup over the previous version.

---

### [`17-parallel-histogram-hybrid`](./17-parallel-histogram-hybrid)

In this project, I also extended the shared memory parallel histogram kernel (not the previous one) with thread coarsening. The difference is that in this project, the memory access pattern of threads is **sequential**.

Each thread processes a contiguous **chunk** of input characters depending on the coarse factor.
After processing its chunk, the thread jumps to the next chunk in a grid-stride loop.

Profiling revealed this sequential access outperformed the interleaved access from the previous project. This is most likely because for small coarse factor values the cache gets utilized very well.

The hyperparameters in this project were optimized using an automated **grid search** (see `run_search.sh` file). The final kernel achieved another ~2.5x speedup over the interleaved version. Compared to a naive single-threaded CPU implementation, the total speedup is over 1,000x.

---

### [`18-parallel-reduction`](./18-parallel-reduction)

In this project, I implemented an optimized parallel reduction kernel which computes the sum of an array of integers.

Each thread loads a number of input values depending on the coarse factor before storing their sum into the shared memory. The values in shared memory are then reduced using a decreasing stride. The final result of each block is then added to the global sum using a single atomic operation.

This implementation achieves a ~20x speedup over a naive single-threaded CPU implementation which does not require any atomic operations.

---

### [`19-prefix-sum-single-block`](./19-prefix-sum-single-block)

In this project, I implemented two single-block prefix sum (sequential scan) kernels. One uses the **Kogge-Stone** prefix sum algorithm and the other one uses the **Brent-Kung prefix** sum algorithm.

There is a trade-off between the algorithms but it is difficult to analyze it in-depth using NCU given the one block limitation (only up to 1,024 elements).

---

### [`20-prefix-sum-single-block-coarsened`](./20-prefix-sum-single-block-coarsened)

In this project, I extended the previous prefix Kogge-Stone prefix sum kernel by using shared memory and thread coarsening.

In the first part, data is loaded into shared memory and each thread performs prefix sum on a chunk assigned to it.

In the second part, the Kogge-Stone prefix sum algorithm is applied
to the last elements of chunks.

In the third part, each chunk is updated by adding the last element of the previous chunk
and the results are written back to the global memory.

The speedup is very minor given the one-block limitation but this implementation serves as the building block for the next project which works with an arbitrary number of elements.

---

### [`21-prefix-sum-general`](./21-prefix-sum-general)

In this project, I implemented a general prefix sum kernel using the Kogge-Stone algorithm.

This solution uses a two-phase approach. The **up-sweep** phase recursively applies a block-level scan to build a reduction tree of partial sums across the entire input. The **down-sweep** phase adds the partial block sums back down to produce the correct prefix sum for every element.

The first level of the up-sweep reads the entire input array, which is by far the most time-consuming step. However, this pass has perfectly coalesced memory access pattern.

Overall, this complex implementation achieves a ~30x speedup over a naive single-threaded CPU implementation.

---

### [`22-sparse-matrix-vector-coo`](./22-sparse-matrix-vector-coo)

In this project, I implemented sparse matrix-vector multiplication (SpMV) kernel using the **COO** (coordinate) format. This format is convenient to work with but results in a very scattered global memory access for the vector.

Each thread computes the product of one non-zero element of the matrix and atomicAdd is used to prevent race conditions which can read to a potential contention bottleneck.

---

### [`23-sparse-matrix-vector-ell`](./23-sparse-matrix-vector-ell)

In this project, I implemented sparse matrix-vector multiplication (SpMV) kernel using the **ELL** (ELLPACK) format. With this format, each thread writes to its own unique row so no atomic operations are needed.

The padding and the column-major ordering of elements results in coalesced memory access patterns. However, during the later iterations, a lot of zero elements are read which do not contribute to the result.

The result is a ~2x speedup over the previous COO implementation.

---

### [`24-bfs-naive`](./24-bfs-naive/)

In this project, I implemented a naive parallel **BFS** algorithm. The **CSR** format is used for the graph.

The host launches a separate grid for each level of the BFS. It iterates over all vertices, which is inefficient but works quite will if the graph is reasonably dense.

In my case, even this naive approach achieved ~80x speedup over the naive single-threaded CPU implementation.

## License

This project is licensed under the MIT License.

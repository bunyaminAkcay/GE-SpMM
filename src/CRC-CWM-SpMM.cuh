/*

    Coalesced Row Caching Coarse-grained Warp Merging Sparse Matrix - Dense Matrix Multiplication

    Reference:
    G. Huang, G. Dai, Y. Wang, and H. Yang, “GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph
    Neural Networks,” IEEE Xplore, Nov. 01, 2020. https://ieeexplore.ieee.org/abstract/document/9355302/ (accessed Sep. 21, 2022).

*/
#include <iostream>
#include <cmath>

/*
    COARSENING_FACTOR is a templete variable. It is
    good because it is compile time constant so we
    can unroll the for loop.
*/
template <typename T, int COARSENING_FACTOR>
__global__ void CRC_CWM_SpMM_Kernel(int m, int n, int *d_A_rowPtr, int *d_A_colIds, T *d_A_values, T *d_B, T *d_C)
{
    extern __shared__ int sharedMemory[];

    int *smColInd = sharedMemory;
    T *smValues = (T *)&sharedMemory[blockDim.y * 32];

    int smOffset = threadIdx.y * 32;
    int threadId = threadIdx.y * 32 + threadIdx.x;

    int rowId = blockDim.y * blockIdx.x + threadIdx.y;

    if (rowId < m)
    {
        int colId = blockIdx.y * 32 * COARSENING_FACTOR + threadIdx.x;
        int rowStart = d_A_rowPtr[rowId];
        int rowEnd = d_A_rowPtr[rowId + 1];
        int ptr = rowStart + threadIdx.x;

        T sums[COARSENING_FACTOR] = {0};

        #pragma unroll
        for (int ci = 0; ci < COARSENING_FACTOR; ci++)
        {
            sums[COARSENING_FACTOR] = static_cast<T>(0);
        }

        for (int i = rowStart; i < rowEnd; i += 32)
        {
            if (ptr < rowEnd)
            {
                smValues[threadId] = d_A_values[ptr];
                smColInd[threadId] = n * d_A_colIds[ptr];
            }
            __syncwarp();
            ptr += 32;

            int loopSize = min(32, rowEnd - i);

            for (int kk = 0; kk < loopSize; kk++)
            {
                T val = smValues[smOffset + kk];
                int offset = smColInd[smOffset + kk] + colId;

                #pragma unroll
                for (int ci = 0; ci < COARSENING_FACTOR; ci++)
                {
                    int col = colId + 32 * ci;
                    if (col < n)
                    {
                        sums[ci] += val * d_B[offset + 32 * ci];
                    }
                }
            }
        }

        #pragma unroll
        for (int ci = 0; ci < COARSENING_FACTOR; ci++)
        {
            int col = colId + 32 * ci;
            if (col < n)
            {
                d_C[rowId * n + col] = sums[ci];
            }
        }
    }
}

template <typename T>
void CRC_CWM_SpMM(int *h_A_rowPtr, int *h_A_colIds, T *h_A_values, T *h_B, T *h_C, T *h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta)
{
    T *d_B, *d_C;
    int *d_A_colIds, *d_A_rowPtr;
    T *d_A_values;

    int const coarseningFactor = 4;

    size_t const warpSize = 32;

    size_t const blockWidth = warpSize;
    size_t const blockRow = 8;

    dim3 const dimBlock(blockWidth, blockRow, 1U);
    dim3 const dimGrid((m + blockRow - 1) / blockRow, (n + (blockWidth * coarseningFactor) - 1) / (blockWidth * coarseningFactor), 1U);
    int sharedMemorySize = 32 * blockRow * (sizeof(int) + sizeof(T));

    cudaMalloc((void **)&d_B, k * n * sizeof(T));
    cudaMalloc((void **)&d_C, m * n * sizeof(T));
    cudaMalloc((void **)&d_A_colIds, nnz * sizeof(int));
    cudaMalloc((void **)&d_A_rowPtr, (m + 1) * sizeof(int));
    cudaMalloc((void **)&d_A_values, nnz * sizeof(T));

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "CUDA memory allocation failed\n");
        goto cleanup;
    }

    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_colIds, h_A_colIds, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rowPtr, h_A_rowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(T), cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "CUDA memory copy failed\n");
        goto cleanup;
    }

    CRC_CWM_SpMM_Kernel<T, coarseningFactor><<<dimGrid, dimBlock, sharedMemorySize, 0>>>(m, n, d_A_rowPtr, d_A_colIds, d_A_values, d_B, d_C);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

cleanup:
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);
}
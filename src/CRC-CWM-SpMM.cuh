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
__global__ void CRC_CWM_SpMM_Kernel(size_t m, size_t n, int *d_A_rowPtr, int *d_A_colIds, T *d_A_values, T *d_B, T *d_C, T alpha, T beta)
{

    int const warpSize = 32;

    int const i = blockIdx.x;  // tb_id
    
    int const j = threadIdx.x % warpSize; // j is laneId

    int const tileId = threadIdx.x / warpSize;
    int const tilePtr = tileId * warpSize * COARSENING_FACTOR;

    int const rowStart = d_A_rowPtr[i];
    int const rowEnd = d_A_rowPtr[i + 1];

    __shared__ T smValues[32];
    __shared__ int smColIds[32];


    T results[COARSENING_FACTOR];

    //init results array to zero
    for (int i = 0; i < COARSENING_FACTOR; i++)
    {
        results[i] = 0;
    }

    for (int ptr = rowStart; ptr < rowEnd; ptr += warpSize)
    {
        if (ptr + j < rowEnd)
        {
            smColIds[j] = d_A_colIds[ptr + j];
            smValues[j] = d_A_values[ptr + j];
        }
        __syncwarp();

        for (int kk = 0; kk < warpSize; kk++)
        {
            int k = smColIds[kk];

            #pragma unroll
            for (int ri = 0; ri < COARSENING_FACTOR; ri++)
            {
                if (tilePtr + (j + ri * warpSize) < n)
                {
                    results[ri] += smValues[kk] * d_B[k * n + tilePtr + (j + ri * warpSize)];
                }
            }   
        }
    }

    #pragma unroll
    for (size_t ri = 0; ri < COARSENING_FACTOR; ri++)
    {
        if (tilePtr + (j + ri * warpSize) < n)
        {
            d_C[i * n + tilePtr + (j + warpSize * ri)] = alpha * results[ri] + beta * d_C[i * n + tilePtr + (j + warpSize * ri)];
        }
    }
}

template <typename T>
void CRC_CWM_SpMM(int *h_A_rowPtr, int *h_A_colIds, T *h_A_values, T *h_B, T *h_C, T *h_D, size_t m, size_t k, size_t const n, size_t nnz, T alpha, T beta)
{
    T *d_B, *d_C;
    int *d_A_colIds, *d_A_rowPtr;
    T *d_A_values;

    cudaMalloc((void **)&d_B, k * n * sizeof(T));
    cudaMalloc((void **)&d_C, m * n * sizeof(T));
    cudaMalloc((void **)&d_A_colIds, nnz * sizeof(int));
    cudaMalloc((void **)&d_A_rowPtr, (m + 1) * sizeof(int));
    cudaMalloc((void **)&d_A_values, nnz * sizeof(T));

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "CUDA memory allocation failed\n");
        return;
    }

    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_colIds, h_A_colIds, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rowPtr, h_A_rowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(T), cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess)
    {
        fprintf(stderr, "CUDA memory copy failed\n");
        return;
    }

    size_t const warpSize = 32;
    size_t const warpCountInRow = (n + warpSize - 1) / warpSize;
    
    int const coarseningFactor = 4;

    int tileCount = ( warpCountInRow + coarseningFactor - 1) /  coarseningFactor;

    int const tileSize = warpSize;

    dim3 const dimBlock(tileSize * tileCount, 1U, 1U);
    dim3 const dimGrid(m, 1U, 1U);

    CRC_CWM_SpMM_Kernel<T, coarseningFactor><<<dimGrid, dimBlock>>>(m, n, d_A_rowPtr, d_A_colIds, d_A_values, d_B, d_C, alpha, beta);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);
}
/*

    Coalesced Row Caching Coarse-grained Warp Merging Sparse Matrix - Dense Matrix Multiplication

    Reference:
    G. Huang, G. Dai, Y. Wang, and H. Yang, “GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph 
    Neural Networks,” IEEE Xplore, Nov. 01, 2020. https://ieeexplore.ieee.org/abstract/document/9355302/ (accessed Sep. 21, 2022).

*/
#include <iostream>

template<typename T, int BLOCK_SIZE>
__global__ void CRC_CWM_SpMM_Kernel(size_t m, size_t n, int* d_A_rowPtr, int* d_A_colIds, T* d_A_values, T* d_B, T* d_C, T alpha, T beta){
    
    int const warpSize = 32;

    int const i = blockIdx.x;   //tb_id
    int const j = threadIdx.x;  //tid

    int const laneId = j % warpSize;
    int const smBase = j - laneId;

    int const rowStart = d_A_rowPtr[i];
    int const rowEnd = d_A_rowPtr[i+1];

    __shared__ T smValues[BLOCK_SIZE];
    __shared__ int smColIds[BLOCK_SIZE];

    T result1 = 0;
    T result2 = 0;
    
    //

    for (int ptr = rowStart; ptr < rowEnd; ptr += warpSize)
    {
        if (ptr + laneId < rowEnd)
        {
            smColIds[j] = d_A_colIds[ptr + laneId];
            smValues[j] = d_A_values[ptr + laneId];
        }
        __syncwarp();
        
        for (int kk = 0; kk < warpSize; kk++)
        {
            int k = smColIds[smBase + kk];
            
            result1 += smValues[smBase + kk] * d_B[k * n + j];
            result2 += smValues[smBase + kk] * d_B[k * n + j + warpSize];
        }
    }

    if (j < n)
    {
        d_C[i * n + j] = alpha * result1 + beta * d_C[i * n + j];
    }

    if (j + warpSize < n)
    {
        d_C[i * n + j + warpSize] = alpha * result2 + beta * d_C[i * n + j + warpSize];
    }
    
}


template<typename T>
void CRC_CWM_SpMM(int* h_A_rowPtr, int* h_A_colIds, T* h_A_values, T* h_B, T* h_C, T* h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta){
    T *d_B, *d_C;
    int *d_A_colIds, *d_A_rowPtr;
    T *d_A_values; 


    cudaMalloc((void**) &d_B, k * n * sizeof(T));
    cudaMalloc((void**) &d_C, m * n * sizeof(T));
    cudaMalloc((void**) &d_A_colIds, nnz * sizeof(int));
    cudaMalloc((void**) &d_A_rowPtr, (m+1) * sizeof(int));
    cudaMalloc((void**) &d_A_values, nnz * sizeof(T));

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed\n");
        return;
    }

    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_colIds, h_A_colIds, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rowPtr, h_A_rowPtr, (m+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(T), cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory copy failed\n");
        return;
    }

    size_t const blockHeight = 4;
    size_t const warpSize = 32;

    int const blockSize = warpSize * blockHeight; 
    dim3 const dimBlock(blockSize, 1U, 1U);
    dim3 const dimGrid(m, 1U, 1U);

    CRC_CWM_SpMM_Kernel<T, blockSize><<<dimGrid, dimBlock>>>(m, n, d_A_rowPtr, d_A_colIds, d_A_values, d_B, d_C, alpha, beta);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);
}
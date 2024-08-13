/*

    Coalesced Row Caching Sparse Matrix - Dense Matrix Multiplication

    Reference:
    G. Huang, G. Dai, Y. Wang, and H. Yang, “GE-SpMM: General-Purpose Sparse Matrix-Matrix Multiplication on GPUs for Graph
    Neural Networks,” IEEE Xplore, Nov. 01, 2020. https://ieeexplore.ieee.org/abstract/document/9355302/ (accessed Sep. 21, 2022).

*/

template <typename T>
__global__ void CRCSpMMKernel(size_t m, size_t n, int *d_A_rowPtr, int *d_A_colIds, T *d_A_values, T *d_B, T *d_C)
{
    extern __shared__ int sharedMemory[];

    int *smColInd = sharedMemory;
    T *smValues = (T *)&sharedMemory[blockDim.y * 32];

    int smOffset = threadIdx.y * 32;
    int threadId = threadIdx.y * 32 + threadIdx.x;

    int rowId = blockDim.y * blockIdx.x + threadIdx.y;

    if (rowId < m)
    {
        int colId = blockIdx.y * 32 + threadIdx.x;
        int rowStart = d_A_rowPtr[rowId];
        int rowEnd = d_A_rowPtr[rowId + 1];
        int ptr = rowStart + threadIdx.x;
        int offset;
        T sum = 0;

        for (int i = rowStart; i < rowEnd; i += 32)
        {
            if (ptr < rowEnd)
            {
                smValues[threadId] = d_A_values[ptr];
                smColInd[threadId] = d_A_colIds[ptr];
            }
            __syncwarp();
            
            ptr += 32;

            int loopSize = min(32, rowEnd - i);

            for (int kk = 0; kk < loopSize; kk++)
            {
                offset = n * smColInd[smOffset + kk] + colId;
                if (colId < n)
                {
                    sum += smValues[smOffset + kk] * d_B[offset];
                }
            }
        }
        if (colId < n)
        {
            d_C[rowId * n + colId] = sum;
        }
    }
}

template <typename T>
void CRCSpMM(int *h_A_rowPtr, int *h_A_colIds, T *h_A_values, T *h_B, T *h_C, T *h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta)
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

    size_t const blockWidth = warpSize;
    size_t const blockRow = 8;

    dim3 const dimBlock(blockWidth, blockRow, 1U);
    dim3 const dimGrid((m + blockRow - 1) / blockRow, (n + blockWidth - 1) / blockWidth, 1U);

    size_t sharedMemorySize = 32 * blockRow * (sizeof(int) + sizeof(T));

    CRCSpMMKernel<T><<<dimGrid, dimBlock, sharedMemorySize, 0>>>(m, n, d_A_rowPtr, d_A_colIds, d_A_values, d_B, d_C);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);
}
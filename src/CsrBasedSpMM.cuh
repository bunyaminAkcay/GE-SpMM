template<typename T>
__global__ void CsrBasedSpMMKernel(size_t m, size_t n, int* d_A_rowPtr, int* d_A_colIds, T* d_A_values, T* d_B, T* d_C, T alpha, T beta){

    size_t const colIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (rowIdx < m && colIdx < n){

        T result = static_cast<T>(0);

        int start = d_A_rowPtr[rowIdx];
        int end = d_A_rowPtr[rowIdx+1];
        
        for (int i = start; i < end; i++)
        {
            int col = d_A_colIds[i];
            result += d_A_values[i] * d_B[ col * n + colIdx];
        }

        d_C[rowIdx * n + colIdx] = alpha * result + beta * d_C[rowIdx * n + colIdx];
    }
}


template<typename T>
void CsrBasedSpMM(int* h_A_rowPtr, int* h_A_colIds, T* h_A_values, T* h_B, T* h_C, T* h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta){
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

    size_t const blockWidth = 16;

    dim3 const dimBlock(blockWidth, blockWidth, 1U);
    dim3 const dimGrid((n + blockWidth -1)/blockWidth, (m + blockWidth -1)/blockWidth, 1U);

    CsrBasedSpMMKernel<T><<<dimGrid, dimBlock>>>(m, n, d_A_rowPtr, d_A_colIds, d_A_values, d_B, d_C, alpha, beta);

    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);


    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);
}
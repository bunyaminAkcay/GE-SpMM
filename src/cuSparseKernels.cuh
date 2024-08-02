#include <cuda_runtime.h>
#include <cusparse.h>

template<typename T>
void cuSparseSpMMwithCOO(int* h_A_rowIds, int* h_A_colIds, T* h_A_values, T* h_B, T* h_C, T* h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta){
    
    T *d_B, *d_C;
    int *d_A_colIds, *d_A_rowIds;
    T *d_A_values; 

    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cudaMalloc((void**) &d_B, k * n * sizeof(T));
    cudaMalloc((void**) &d_C, m * n * sizeof(T));
    cudaMalloc((void**) &d_A_colIds, nnz * sizeof(int));
    cudaMalloc((void**) &d_A_rowIds, nnz * sizeof(int));
    cudaMalloc((void**) &d_A_values, nnz * sizeof(T));

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed\n");
        return;
    }

    cudaMemcpy(d_B, h_B, k * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_colIds, h_A_colIds, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_rowIds, h_A_rowIds, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(T), cudaMemcpyHostToDevice);

    if (cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "CUDA memory copy failed\n");
        return;
    }

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cusparseCreateCoo(&matA, m, k, nnz, d_A_rowIds, d_A_colIds, d_A_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnMat(&matB, k, n, n, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, m, n, n, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW);

    size_t bufferSize;
    void *dBuffer;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Perform SpMM
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    // Free workspace
    

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaFree(dBuffer);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowIds);
    cudaFree(d_A_values);

}

template<typename T>
void cuSparseSpMMwithCSR(int* h_A_rowPtr, int* h_A_colIds, T* h_A_values, T* h_B, T* h_C, T* h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta){
    
    T *d_B, *d_C;
    int *d_A_colIds, *d_A_rowPtr;
    T *d_A_values; 

    cusparseHandle_t handle;
    cusparseCreate(&handle);

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

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    cudaDataType_t elementDataType;
    if (sizeof(T) == 4)
    {
        elementDataType = CUDA_R_32F;
    }
    else if (sizeof(T) == 8)
    {
        elementDataType = CUDA_R_64F;
    }
    else
    {
        printf("Unknown datatype. size = %lu\n", sizeof(T));
        exit(-1);
    }
    
    
    

    cusparseCreateCsr(&matA, m, k, nnz, d_A_rowPtr, d_A_colIds, d_A_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, elementDataType);

    cusparseCreateDnMat(&matB, k, n, n, d_B, elementDataType, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matC, m, n, n, d_C, elementDataType, CUSPARSE_ORDER_ROW);

    size_t bufferSize;
    void *dBuffer;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, elementDataType, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Perform SpMM
    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC, elementDataType, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);


    cudaMemcpy(h_D, d_C, m * n * sizeof(T), cudaMemcpyDeviceToHost);

    // Free workspace
    

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    cudaFree(dBuffer);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_colIds);
    cudaFree(d_A_rowPtr);
    cudaFree(d_A_values);

}
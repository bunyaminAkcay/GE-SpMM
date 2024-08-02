#include "MatrixParser.h"
#include "MatrixHelper.h"
#include "cuSparseKernels.cuh"
#include "CsrBasedSpMM.cuh"
#include "CRCSpMM.cuh"
#include <string>

typedef double element_t;

template<typename T>
using KernelType = void (*)(int* h_A_rowPtr, int* h_A_colIds, T* h_A_values, T* h_B, T* h_C, T* h_D, size_t m, size_t k, size_t n, size_t nnz, T alpha, T beta);

struct testMatrix
{
    std::string fileName;
    bool symmetrical;
};


testMatrix matrices[] = {testMatrix{"494_bus.mtx", true}, testMatrix{"662_bus.mtx", true}, testMatrix{"1138_bus.mtx", true}, testMatrix{"abb313.mtx", false}, testMatrix{"arc130.mtx", false}, testMatrix{"ash219.mtx", false}, testMatrix{"b1_ss.mtx", false}, testMatrix{"bcspwr04.mtx", true}, testMatrix{"bcsstk01.mtx", true}, testMatrix{"blckhole.mtx", true}, testMatrix{"can_24.mtx", true}, testMatrix{"dwt_607.mtx", true}, testMatrix{"eris1176.mtx", true} };


void testSpMM(const testMatrix &testMatrix, KernelType<element_t> kernel,  double tolerance, bool printResult = false){
        
        CsrMatrixParser<element_t> csrMatrixParser("matrices/"+testMatrix.fileName, testMatrix.symmetrical);
        //csrMatrixParser.saveSparseMatrixAsPPM3Image("matrixImages/"+testMatrix.fileName);
        
        size_t m = csrMatrixParser.getRowCount();
        size_t k = csrMatrixParser.getColCount();
        size_t n = m;
        size_t nnz = csrMatrixParser.getNNZ();

        element_t alpha = static_cast<element_t>(1);
        element_t beta = static_cast<element_t>(0);

        int* h_A_rowPtr = csrMatrixParser.rowPtr;
        int* h_A_colIds = csrMatrixParser.colIds;
        element_t* h_A_values = csrMatrixParser.values; 

        element_t *h_B = (element_t *)malloc(k * n * sizeof(element_t));
        element_t *h_C = (element_t *)malloc(m * n * sizeof(element_t));
        element_t *h_D_cuSparse = (element_t *)malloc(m * n * sizeof(element_t));
        element_t *h_D_csrBased = (element_t *)malloc(m * n * sizeof(element_t));
        
        MatrixHelper<element_t>::initRandomDenseMatrix(h_B, k, n);
        MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
        MatrixHelper<element_t>::initZeroMatrix(h_D_cuSparse, m, n);
        
        //perform cuSparse
        cuSparseSpMMwithCSR<element_t>(h_A_rowPtr, h_A_colIds, h_A_values, h_B, h_C, h_D_cuSparse, m, k, n, nnz, alpha, beta);
        
        
        if(printResult && csrMatrixParser.sparseMatrixToClassicMatrix()){
            MatrixHelper<element_t>::printResult(m, n, k, csrMatrixParser.getClassicMatrix(), h_B, h_C, h_D_cuSparse, true);
        }

        //perform Csr based SpMM
        MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
        MatrixHelper<element_t>::initZeroMatrix(h_D_csrBased, m, n);

        kernel(h_A_rowPtr, h_A_colIds, h_A_values, h_B, h_C, h_D_csrBased, m, k, n, nnz, alpha, beta);

        if(printResult && csrMatrixParser.sparseMatrixToClassicMatrix()){
            MatrixHelper<element_t>::printResult(m, n, k, csrMatrixParser.getClassicMatrix(), h_B, h_C, h_D_csrBased, true);
        }

        free(h_B);
        free(h_C);
        
        
        int wrongElementCount = 0;
        //compare results
        for (int rowId = 0; rowId < m; rowId++)
        {
            for (int colId = 0; colId < n; colId++)
            {
                if (abs(h_D_cuSparse[rowId * n + colId] - h_D_csrBased[rowId * n + colId]) > tolerance)
                {
                    wrongElementCount++;
                }
                
            }
        }
        
        float correct = m * n - wrongElementCount;
        float total = m * n;
        printf("Correctness: %.4f%%", (100*correct/total));
        
        correct == total ? printf("\tTEST PASSED") : printf("\tTEST FAILED");
        
        printf("\tmatrix size = %lux%lu\n", m, n);

        free(h_D_cuSparse);
        free(h_D_csrBased);
}

int main()
{
    double tolerance = 0.0001;
    
    printf("Tolerance:\t %.8lf\n", tolerance);
    
    printf("\nCsr based SpMM test results\n======================\n");
    
    for(const testMatrix &testMatrix : matrices){
        testSpMM(testMatrix, CsrBasedSpMM, tolerance);
    }

    printf("\nCRC SpMM test results\n======================\n");
    
    for(const testMatrix &testMatrix : matrices){
        testSpMM(testMatrix, CRCSpMM, tolerance);
    }

    //testSpMM(matrices[8], CRCSpMM, tolerance, true);

    return 0;
}
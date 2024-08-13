#include "../MatrixParser.h"
#include "../MatrixHelper.h"

#include "../CsrBasedSpMM.cuh"

#include <string>

typedef double element_t;

int main()
{
    bool printResult = false;
    CsrMatrixParser<element_t> csrMatrixParser("matrices/blckhole.mtx", true);
    
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
    element_t *h_D_csrBased = (element_t *)malloc(m * n * sizeof(element_t));
    
    MatrixHelper<element_t>::initRandomDenseMatrix(h_B, k, n);
    MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
    
    //perform Csr based SpMM
    MatrixHelper<element_t>::initZeroMatrix(h_C, m, n);
    MatrixHelper<element_t>::initZeroMatrix(h_D_csrBased, m, n);

    CsrBasedSpMM(h_A_rowPtr, h_A_colIds, h_A_values, h_B, h_C, h_D_csrBased, m, k, n, nnz, alpha, beta);

    if(printResult && csrMatrixParser.sparseMatrixToClassicMatrix()){
        MatrixHelper<element_t>::printResult(m, n, k, csrMatrixParser.getClassicMatrix(), h_B, h_C, h_D_csrBased, true);
    }

    free(h_B);
    free(h_C);
    free(h_D_csrBased);

    return 0;
}
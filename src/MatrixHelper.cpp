#include "MatrixHelper.h"

#include <cstdlib>
#include <iostream>

template<typename T>
void MatrixHelper<T>::initRandomDenseMatrix(T *matrix, size_t const rowCount, size_t const colCount)
{
    for (size_t row = 0; row < rowCount; row++)
    {
        for (size_t col = 0; col < colCount; col++)
        {
            matrix[row * colCount + col] = rand() / static_cast<T>(RAND_MAX);
        }
    }
}

template<typename T>
void MatrixHelper<T>::initZeroMatrix(T *matrix, size_t const rowCount, size_t const colCount)
{
    for (size_t row = 0; row < rowCount; row++)
    {
        for (size_t col = 0; col < colCount; col++)
        {
            matrix[row * colCount + col] = 0;
        }
    }
}

template<typename T>
void MatrixHelper<T>::printResult(size_t const m, size_t const n, size_t const k, T const* A, T const* B, T const* C, T const* D, bool justMatrixD)
{
    size_t const matrixCount = 4;
    T const* matrixReferences[matrixCount] = {A, B, C, D};

    char matrixNames[matrixCount] = {'A', 'B', 'C', 'D'};
    size_t matrixDimensions[matrixCount][2] = { {m, k}, {k, n}, {m, n}, {m, n}};

    
    for (size_t k = 0; k < matrixCount; k++)
    {
        if (justMatrixD)
        {
            k = matrixCount -1;
        }
        
        printf("%c [\n\n\t", matrixNames[k]);
        for (size_t j = 0; j < matrixDimensions[k][0]; j++)
        {
            for (size_t i = 0; i < matrixDimensions[k][1]; i++)
            {
                std::cout << matrixReferences[k][matrixDimensions[k][1] * j + i] << " ";
                //printf("%f ", matrixReferences[k][matrixDimensions[k][1] * j + i]);
            }
            printf("\n\t");
        }
        printf("\n]\n");
    }
}

template class MatrixHelper<int>;
template class MatrixHelper<float>;
template class MatrixHelper<double>;
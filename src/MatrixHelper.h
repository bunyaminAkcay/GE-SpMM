#include <cstdio>

template<typename T>
class MatrixHelper{
    public:
        static void initRandomDenseMatrix(T *matrix, size_t const rowCount, size_t const colCount);
        static void initZeroMatrix(T *matrix, size_t const rowCount, size_t const colCount);
        static void printResult(size_t const m, size_t const n, size_t const k, T const* A, T const* B, T const* C, T const* D, bool justMatrixD = false);
};
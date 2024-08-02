#include <string>

enum Order{rowMajor, colMajor};

template<typename T>
class MatrixParser{

    public:
        size_t getRowCount();
        size_t getColCount();
        size_t getNNZ();
        ~MatrixParser();
        bool sparseMatrixToClassicMatrix();//return true if success
        T* getClassicMatrix();
        void sort(Order order);
        bool saveSparseMatrixAsPPM3Image(std::string fileName);

    protected:
        bool symmetrical;
        int *_rowIds, *_colIds;
        int rowCount, colCount, nnz;
        T maxValue = 0;
        T *_values;
        T *classicMatrix = new T[1];
        void printCoordinates();
        void readMatrixFromFile(std::string fileName);
        
    private:
        bool classicMatrixAllocated = false;
        size_t maxMatrixSizeToConvertNormalMatrix = 2048;
};

template<typename T>
class CooMatrixParser : public MatrixParser<T>{

    public:
        int *rowIds, *colIds;
        T *values;
        void printCOO();
        CooMatrixParser(std::string fileName, bool symmetrical, Order order);
        ~CooMatrixParser();
};

template<typename T>
class CsrMatrixParser : public MatrixParser<T>{
    public:
        int *rowPtr, *colIds;
        T* values;
        void printCSR();
        CsrMatrixParser(std::string fileName, bool symmetrical);
        ~CsrMatrixParser();
};

template<typename T>
class CscMatrixParser : public MatrixParser<T>{
    public:
        int *rowIds, *colPtr;
        T* values;
        void printCSC();
        CscMatrixParser(std::string fileName, bool symmetrical);
        ~CscMatrixParser();
};
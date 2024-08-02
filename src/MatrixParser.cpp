#include "MatrixParser.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <cassert>
#include <cstring>

template<typename T>
void MatrixParser<T>::readMatrixFromFile(std::string fileName){
    
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        exit(-1);
    }

    std::string line;
    
    while (std::getline(infile, line)) {
        if (line[0] != '%' && !line.empty()) {
            break;
        }
    }

    std::istringstream iss(line);
    
    iss >> rowCount >> colCount >> nnz;

    if (symmetrical)
    {
        // actually for symmetrical case, nnz is equals 2*nnz - nnz_on_diagonal 
        _rowIds = new int[2 * nnz];
        _colIds = new int[2 * nnz];
        _values = new T [2 * nnz];
    }
    else{
        _rowIds = new int[nnz];
        _colIds = new int[nnz];
        _values = new T [nnz];
    }
    
    bool hasValue = false;

    std::streampos pos = infile.tellg();
    std::getline(infile, line);
    std::istringstream test_iss(line);
    size_t testRowId, testColId;
    T testValue;

    int index = 0;

    if (test_iss >> testRowId >> testColId >> testValue) {
        hasValue = true;
        _rowIds[index] = testRowId - 1;
        _colIds[index] = testColId - 1;
        _values[index] = testValue;
        maxValue = abs(testValue);
        index++;
        
    } else {
        _rowIds[index] = testRowId - 1;
        _colIds[index] = testColId - 1;
        maxValue = 1;
        index++;
    }

    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);

        size_t rowId, colId;
        T value;
        iss >> rowId >> colId;
        _rowIds[index] = rowId - 1;
        _colIds[index] = colId - 1;
        if (hasValue) {
            iss >> value;
            
            if (abs(value) > maxValue)
            {
                maxValue = abs(value);
            }
            _values[index] = value;
        }
        else {
            _values[index] = static_cast<T>(1);
        }

        index++;

        if (symmetrical)
        {
            _rowIds[index] = colId - 1;
            _colIds[index] = rowId - 1;
            

            if (hasValue) {
                
                _values[index] = value;
            }
            else {
                _values[index] = static_cast<T>(1);
            }
            index++;   
        }
        
    }

    if (!symmetrical)
    {
        assert(nnz == index);
    }

    nnz = index;
    
    if (symmetrical)
    {
        int* new_rowIds = new int[nnz];
        int* new_colIds = new int[nnz];
        T* new_values = new T[nnz];
        memcpy(new_rowIds, _rowIds, nnz * sizeof(int));
        memcpy(new_colIds, _colIds, nnz * sizeof(int));
        memcpy(new_values, _values, nnz * sizeof(T));

        delete[] _rowIds;
        delete[] _colIds;
        delete[] _values;

        _rowIds = new_rowIds;
        _colIds = new_colIds;
        _values = new_values;
    }
    
    infile.close();
}

template<typename T>
size_t MatrixParser<T>::getRowCount(){
    return rowCount;
}

template<typename T>
size_t MatrixParser<T>::getColCount(){
    return colCount;
}

template<typename T>
size_t MatrixParser<T>::getNNZ(){
    return nnz;
}

template<typename T>
T* MatrixParser<T>::getClassicMatrix(){
    return classicMatrix;
}

template<typename T>
void MatrixParser<T>::printCoordinates(){
    for (size_t i = 0; i < nnz; i++)
    {
        std::cout << _rowIds[i] << " " << _colIds[i] << " " << _values[i] << "\n";
        //printf("%d %d %f\n", _rowIds[i], _colIds[i], _values[i]);
    }
    
}

template<typename T>
void MatrixParser<T>::sort(Order order){
    
    int *primary, *secondary;

    if (order == Order::rowMajor)
    {
        primary = _rowIds;
        secondary = _colIds;
    }
    else if (order == Order::colMajor)
    {
        primary = _colIds;
        secondary = _rowIds;
    }
    
    //sort for secondary

    bool sortedForTarget = false;
    int* target = secondary;

    for (size_t i = 0; i < 2; i++)
    {
        while (!sortedForTarget)
        {
            sortedForTarget = true;
            for (size_t i = 0; i < nnz - 1; i++)
            {
                if(target[i] > target[i+1]){
                    sortedForTarget = false;
                    int temp_row = _rowIds[i];
                    int temp_col = _colIds[i];
                    T temp_val = _values[i];
                    
                    _rowIds[i] = _rowIds[i+1];
                    _colIds[i] = _colIds[i+1];
                    _values[i] = _values[i+1];

                    _rowIds[i+1] = temp_row;
                    _colIds[i+1] = temp_col;
                    _values[i+1] = temp_val;

                }
            }
        }
        sortedForTarget = false;
        target = primary;
    }
}

template<typename T>
bool MatrixParser<T>::sparseMatrixToClassicMatrix(){

    if (classicMatrixAllocated)
    {
        return true;
    }
    

    if (rowCount > maxMatrixSizeToConvertNormalMatrix || colCount > maxMatrixSizeToConvertNormalMatrix)
    {
        printf("Matrix is too big to store as classic way. Max matrix size is %d", int(maxMatrixSizeToConvertNormalMatrix) );
        return false;
    }
    

    delete classicMatrix;
    classicMatrix = new T[rowCount * colCount];
    std::fill(classicMatrix, classicMatrix + rowCount * colCount, T{});

    classicMatrixAllocated = true;
    
    for (int nzi = 0; nzi < nnz; nzi++)
    {
        classicMatrix[_rowIds[nzi] * colCount + _colIds[nzi]] = _values[nzi];
    }

    return true;
}



template<typename T>
MatrixParser<T>::~MatrixParser(){

    free(_rowIds);
    free(_colIds);
    free(_values);
    free(classicMatrix);
}

template<typename T>
CooMatrixParser<T>::CooMatrixParser(std::string fileName, bool symmetrical, Order order){
    this->symmetrical = symmetrical;
    this->readMatrixFromFile(fileName);
    this->sort(order);
    rowIds = this->_rowIds;
    colIds = this->_colIds;
    values = this->_values;
}

template<typename T>
void CooMatrixParser<T>::printCOO(){
    this->printCoordinates();
}


template<typename T>
bool MatrixParser<T>::saveSparseMatrixAsPPM3Image(std::string fileName){

    if (!this->sparseMatrixToClassicMatrix())
    {
        printf("Matrix is too big to save as ppm.");
        return false;
    }
    
    std::ofstream ppm3ImageFile;
    ppm3ImageFile.open ( fileName + ".ppm");

    ppm3ImageFile << "P3\n" << this->rowCount << " " << this->colCount << "\n" << 255 << "\n";

    for (size_t j = 0; j < this->rowCount; j++)
    {
        for (size_t i = 0; i < this->colCount; i++)
        {
            T value = this->classicMatrix[j * this->colCount + i];
            if (value == static_cast<T>(0))
            {
                ppm3ImageFile << "255 255 255 ";
            }
            else
            {
                float normalizedValue = float(abs(value))/this->maxValue;
                float sigmoid = 1 / (1 + exp(-10*normalizedValue));
                sigmoid = 2 * (sigmoid - 0.5);
                uint r = 68 + sigmoid * 185;
                uint g = 1 + sigmoid * 230;
                uint b = 84 - sigmoid * 48;
                ppm3ImageFile << r << " " << g << " " << b << " ";
            }
        }
        ppm3ImageFile << "\n";
    }
    
    ppm3ImageFile.close();
    return true;
}


template<typename T>
CooMatrixParser<T>::~CooMatrixParser(){}

template<typename T>
CsrMatrixParser<T>::CsrMatrixParser(std::string fileName, bool symmetrical){
    this->symmetrical = symmetrical;
    this->readMatrixFromFile(fileName);
    this->sort(Order::rowMajor);

    colIds = this->_colIds;
    values = this->_values;

    rowPtr = new int[this->rowCount + 1];
    std::fill(rowPtr, rowPtr + this->rowCount + 1, 0);
    
    int lastRow = 0;
    int elementCount = 0;
    int rowPtrIndex = 1;
    
    for (size_t i = 0; i < this->nnz; i++)
    {
        
        if (this->_rowIds[i] == lastRow)
        {
            elementCount++;
            continue;
        }

        assert(rowPtrIndex < this->rowCount + 1);

        rowPtr[rowPtrIndex] = elementCount;
        elementCount++;
        rowPtrIndex++;
        lastRow = this->_rowIds[i];
    }
    rowPtr[rowPtrIndex] = elementCount; 
}

template<typename T>
void CsrMatrixParser<T>::printCSR(){

    printf("Row Ptr:\n");
    for (size_t i = 0; i < this->rowCount + 1; i++)
    {
        printf("%d ", rowPtr[i]);
    }

    printf("\nCol Ids:\n");
    for (size_t i = 0; i < this->nnz; i++)
    {
        printf("%d ", colIds[i]);
    }

    printf("\nValues:\n");
    for (size_t i = 0; i < this->nnz; i++)
    {
        std::cout << values[i] << " ";
    }
    printf("\n");
}

template<typename T>
CsrMatrixParser<T>::~CsrMatrixParser(){
    free(rowPtr);
}

template<typename T>
CscMatrixParser<T>::CscMatrixParser(std::string fileName, bool symmetrical){
    this->symmetrical = symmetrical;
    this->readMatrixFromFile(fileName);
    this->sort(Order::colMajor);

    rowIds = this->_rowIds;
    values = this->_values;

    colPtr = new int[this->colCount + 1];
    std::fill(colPtr, colPtr + this->colCount + 1, 0);
    
    int lastCol = 0;
    int elementCount = 0;
    int colPtrIndex = 1;
    
    for (size_t i = 0; i < this->nnz; i++)
    {
        
        if (this->_colIds[i] == lastCol)
        {
            elementCount++;
            continue;
        }

        assert(colPtrIndex < this->colCount + 1);

        colPtr[colPtrIndex] = elementCount;
        elementCount++;
        colPtrIndex++;
        lastCol = this->_colIds[i];
    }
    colPtr[colPtrIndex] = elementCount;
}

template<typename T>
void CscMatrixParser<T>::printCSC(){

    printf("Col Ptr:\n");
    for (size_t i = 0; i < this->colCount + 1; i++)
    {
        printf("%d ", colPtr[i]);
    }

    printf("\nRow Ids:\n");
    for (size_t i = 0; i < this->nnz; i++)
    {
        printf("%d ", rowIds[i]);
    }

    printf("\nValues:\n");
    for (size_t i = 0; i < this->nnz; i++)
    {
        std::cout << values[i] << " ";
    }
    printf("\n");
}

template<typename T>
CscMatrixParser<T>::~CscMatrixParser(){
    free(colPtr);
}


template class MatrixParser<int>;
template class MatrixParser<float>;
template class MatrixParser<double>;

template class CooMatrixParser<int>;
template class CooMatrixParser<float>;
template class CooMatrixParser<double>;

template class CsrMatrixParser<int>;
template class CsrMatrixParser<float>;
template class CsrMatrixParser<double>;

template class CscMatrixParser<int>;
template class CscMatrixParser<float>;
template class CscMatrixParser<double>;
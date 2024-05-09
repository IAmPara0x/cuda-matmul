#ifndef MATRIX_H

#define MATRIX_H

#include <cstddef>
#include <string>

struct Matrix {
    size_t rows;
    size_t cols;
    float *value;
    size_t size;

    // Overload the '==' operator as a member function
    bool operator==(const Matrix &other) const {

        if (other.rows != rows || other.cols != cols) return false;

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                if (std::abs(value[i*rows + j] - other.value[i*rows + j]) > 0.1 ) 
                {
                    printf("%f != %f, (%d, %d), diff=%f\n", value[i*rows + j] , other.value[i*rows + j] , i,j, value[i*rows + j] - other.value[i*rows + j] );
                    return false;
                }

        return true;
    }
};

#define NULL_MATRIX Matrix {0, 0, NULL}

void getDeviceInfo();

Matrix readMat(std::string filename);
Matrix alloc_mat(size_t rows, size_t cols);
void free_mat(Matrix mat);
void print_mat(Matrix mat);
void transpose(Matrix *mat);

#endif

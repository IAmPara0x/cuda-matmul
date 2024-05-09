#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include <stdio.h>
#include <cstdlib>
#include "matrix.h"

using namespace std;

Matrix readMat(string filename) {

    ifstream file(filename); 
    string line;

    vector<string> tokens;
    string token;
    char delimiter = ',';

    bool has_initialized_matrix = false;

    size_t rows, cols;

    Matrix matrix;

    if (!file) {
        cerr << "Unable to open file\n";
        return NULL_MATRIX;
    }

    file.unsetf(ios_base::skipws);
    rows = count( istream_iterator<char>(file), istream_iterator<char>(), '\n');
    file.setf(ios_base::skipws);

    // Reset the file pointer to the beginning of the file
    file.clear(); // Clears any error flags that might be set
    file.seekg(0, std::ios::beg); // Move the file pointer to the beginning


    size_t cur_row = 0;

    while (getline(file, line)) {

        istringstream tokenStream(line);

        while (getline(tokenStream, token, delimiter))
            tokens.push_back(token);

        if (!has_initialized_matrix) {

            cols = tokens.size();
            matrix = alloc_mat(rows, cols);
            has_initialized_matrix = true;
        }

        for (int j = 0; j < cols; j++)
            matrix.value[cur_row * cols + j] = stof(tokens[j]);

        cur_row += 1;
        tokens.clear();
    }

    file.close(); // Close the file

    return matrix;
}

Matrix alloc_mat(size_t rows, size_t cols) {

    Matrix matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.size = rows * cols * sizeof(float);
    matrix.value = (float*)malloc(rows * cols * sizeof(float));
    return matrix;
}

void free_mat(Matrix matrix) {
    free(matrix.value); // Free the array of pointers
}

void print_mat(Matrix matrix) {

    for(int i = 0; i < matrix.rows; i++) {
        for(int j = 0; j < matrix.cols; j++) {
            printf("%8.3f ", matrix.value[i * matrix.cols + j]);
        }
        std::cout << "\n";
    }
}

void transpose(Matrix *matrix) {

    float *value = (float*)malloc(matrix->rows * matrix->cols * sizeof(float));

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            value[j * matrix->cols + i] = matrix->value[i * matrix->cols + j];

    free(matrix->value);
    matrix->value = value;
};

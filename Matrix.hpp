#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <random>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows, cols;

public:
    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    Matrix(int r, int c, const unsigned char* rawData, double scale = 1.0) 
        : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols));
        for (int i = 0; i < rows * cols; i++) {
            data[i / cols][i % cols] = rawData[i] * scale;
        }
    }

    // move, assignment operator
    Matrix(Matrix&& other) noexcept 
        : data(std::move(other.data)), rows(other.rows), cols(other.cols) {
        other.rows = 0;
        other.cols = 0;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            rows = other.rows;
            cols = other.cols;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // copy constructor, assignment
    Matrix(const Matrix& other)
        : data(other.data), rows(other.rows), cols(other.cols) {}

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            data = other.data;
            rows = other.rows;
            cols = other.cols;
        }
        return *this;
    }

    // getters/setters
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
    double get(int r, int c) const { return data[r][c]; }
    void set(int r, int c, double val) { data[r][c] = val; }

    // matrix operations (all return new matrices)
    Matrix multiply(const Matrix& other) const {
        if (cols != other.get_rows()) {
            throw std::invalid_argument("Matrix dimensions don't match for multiplication");
        }
        
        Matrix product = Matrix(rows, other.get_cols());
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.get_cols(); j++) {
                double sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.get(k, j);
                }
                product.set(i, j, sum);
            }
        }
        return product;
    }

    Matrix add(const Matrix& other) const {
        if (rows != other.get_rows() || cols != other.get_cols()) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] + other.get(i, j));
            }
        }
        return result;
    }

    Matrix subtract(const Matrix& other) const {
        if (rows != other.get_rows() || cols != other.get_cols()) {
            throw std::invalid_argument("Matrix dimensions don't match for subtraction");
        }
        
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] - other.get(i, j));
            }
        }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, data[i][j]);
            }
        }
        return result;
    }

    // element-wise multiplication (hadamard product)
    Matrix mult_elements(const Matrix& other) const {
        if (rows != other.get_rows() || cols != other.get_cols()) {
            throw std::invalid_argument("Matrix dimensions don't match for element-wise multiplication");
        }

        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] * other.get(i, j));
            }
        }
        return result;
    }

    // scalar operations (return new matrices)
    Matrix multiply_scalar(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] * scalar);
            }
        }
        return result;
    }

    Matrix add_scalar(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, data[i][j] + scalar);
            }
        }
        return result;
    }

    // randomize (in-place is fine for initialization)
    void randomize(double min = -1.0, double max = 1.0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min, max);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dist(gen);
            }
        }
    }

    // apply a function to entire matrix (returns new matrix)
    Matrix applyFunction(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(i, j, func(data[i][j]));
            }
        }
        return result;
    }

    const std::vector<std::vector<double>>& get_data() const { 
        return data; 
    }

    // print function for debugging
    void print() const {
        std::cout << "Matrix (" << rows << "x" << cols << "):" << std::endl;
        
        for (int i = 0; i < rows; i++) {
            std::cout << "[ ";
            for (int j = 0; j < cols; j++) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << data[i][j];
                if (j < cols - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << std::endl;
    }
};
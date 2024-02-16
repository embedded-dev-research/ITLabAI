#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>

struct tensor_size {
    int height;
    int width;
};

class Matrix {
private:
    tensor_size size;
    std::vector<double> values;
public:
    Matrix();
    Matrix(int height, int width);
    Matrix(tensor_size s);
    Matrix(const Matrix& t);

    tensor_size Get_size() const { return size; }

    double& operator()(int h, int w); // write
    double operator()(int h, int w) const; // read

    friend std::ostream& operator<<(std::ostream& out, const Matrix& t);
};

Matrix::Matrix() {
    size.width = 0;
    size.height = 0;

    values = {};
}

Matrix::Matrix(int height, int width) {
    size.width = width;
    size.height = height;

    values = std::vector<double>(width * height, 0);
}

Matrix::Matrix(tensor_size s) {
    size.width = s.width;
    size.height = s.height;

    values = std::vector<double>(size.width * size.height, 0);
}

Matrix::Matrix(const Matrix& t) {
    tensor_size t_size = t.Get_size();

    size.width = t_size.width;
    size.height = t_size.height;

    values = std::vector<double>(size.width * size.height, 0);
    for (int h = 0; h < size.height; h++) {
        for (int w = 0; w < size.width; w++) {
            (*this)(h, w) = t(h, w);
        }
    }
}

double& Matrix::operator()(int h, int w) {
    return values[h * size.width + w];
} // write
double Matrix::operator()(int h, int w) const {
    return values[h * size.width + w];
} // read

std::ostream& operator<<(std::ostream& out, const Matrix& t) {
    for (int h = 0; h < t.size.height; h++) {
        for (int w = 0; w < t.size.width; w++) {
            std::cout.width(5);
            out << t(h, w) << " ";
        }
        out << std::endl;
    }

    return out;
}

Matrix initial_picture(tensor_size size) {
    srand(time(0));
    Matrix picture(size);

    for (int h = 0; h < size.height; h++) {
        for (int w = 0; w < size.width; w++) {
            picture(h, w) = rand() % 255;
        }
    }

    return picture;
}
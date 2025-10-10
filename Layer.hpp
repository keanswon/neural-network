#pragma once
#include <iostream>
#include "Matrix.hpp"

class Layer {
public:
    Matrix weights;
    Matrix biases;         
    Matrix inputs;         
    Matrix outputs;        
    Matrix weightGradients;
    Matrix biasGradients;
    Matrix activations;
    
    Layer(int inputSize, int outputSize);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& outputGradient, double learningRate);
};

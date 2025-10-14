#pragma once
#include <iostream>
#include "Matrix.hpp"

enum class Activation { RELU, SOFTMAX, NONE };

class Layer {
public:
    Activation activation_type;
    Matrix weights;
    Matrix biases;         
    Matrix inputs;         
    Matrix outputs;        
    Matrix weightGradients;
    Matrix biasGradients;
    Matrix activations;
    
    Layer(int inputSize, int outputSize, Activation act);
    
    Matrix forward(const Matrix& input);
    Matrix backward(const Matrix& outputGradient, double learningRate);
};

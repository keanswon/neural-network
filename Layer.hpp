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
    Matrix weightGradientAccumulator;
    Matrix biasGradientAccumulator;
    
    Layer(int inputSize, int outputSize, Activation act);
    
    Matrix forward(const Matrix& input);
    void update_weights(double learning_rate, int batch_size);
    Matrix backward(const Matrix& outputGradient);
};

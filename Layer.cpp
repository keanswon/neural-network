#include "Layer.hpp"
#include "Helpers.hpp"

Layer::Layer(int inputSize, int outputSize) {
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);
    weights.randomize();
    biases.randomize();
}

Matrix Layer::forward(const Matrix& input) {
    inputs = input;
    
    outputs = weights.multiply(input).add(biases);
    activations = outputs.applyFunction(relu);
    
    return activations;
}

// backprop for one layer
Matrix Layer::backward(const Matrix& outputGradient, double learningRate) {
    Matrix gradient = outputs.applyFunction(relu_deriv).mult_elements(outputGradient);

    Matrix inputsTransposed = inputs.transpose();
    weightGradients = gradient.multiply(inputsTransposed);
    biasGradients = gradient;
    
    weights = weights.subtract(weightGradients.multiply_scalar(learningRate));
    biases = biases.subtract(biasGradients.multiply_scalar(learningRate));
    
    return weights.transpose().multiply(gradient);
}

#include "Layer.hpp"
#include "Helpers.hpp"

Layer::Layer(int inputSize, int outputSize, Activation act) {
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);
    weights.randomize();
    biases.randomize();
    activation_type = act;
}

Matrix Layer::forward(const Matrix& input) {
    inputs = input;
    
    outputs = weights.multiply(input).add(biases);
    
    if (activation_type == Activation::RELU) {
        activations = outputs.applyFunction(relu);

        // int dead_neurons = 0;
        // for (int i = 0; i < activations.get_rows(); i++) {
        //     if (activations.get(i, 0) == 0.0) dead_neurons++;
        // }
        // std::cout << "Dead ReLU neurons: " << dead_neurons << "/" 
        //           << activations.get_rows() << std::endl; // debugging
    } else if (activation_type == Activation::SOFTMAX) {
        activations = softmax(outputs);
    } else {
        activations = outputs;  // No activation
    }
    
    return activations;
}

// backprop for one layer
Matrix Layer::backward(const Matrix& outputGradient, double learningRate) {
    Matrix gradient;

    if (activation_type == Activation::RELU) {
        gradient = outputs.applyFunction(relu_deriv).mult_elements(outputGradient);
    } else {
        gradient = outputGradient;
    }

    Matrix inputsTransposed = inputs.transpose();
    weightGradients = gradient.multiply(inputsTransposed);
    biasGradients = gradient;
    
    weights = weights.subtract(weightGradients.multiply_scalar(learningRate));
    biases = biases.subtract(biasGradients.multiply_scalar(learningRate));

    // double grad_norm = 0.0;
    // for (int i = 0; i < gradient.get_rows(); i++) {
    //     grad_norm += gradient.get(i, 0) * gradient.get(i, 0);
    // }
    // grad_norm = std::sqrt(grad_norm);
    // std::cout << "Gradient norm: " << grad_norm << std::endl; // more debugging
    
    return weights.transpose().multiply(gradient);
}

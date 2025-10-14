#include "Layer.hpp"
#include "Helpers.hpp"

Layer::Layer(int inputSize, int outputSize, Activation act) {
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);
    weights.randomize();
    biases.randomize();
    activation_type = act;

    weightGradientAccumulator = Matrix(outputSize, inputSize);
    biasGradientAccumulator = Matrix(outputSize, 1);
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
Matrix Layer::backward(const Matrix& outputGradient) {
    Matrix gradient;

    if (activation_type == Activation::RELU) {
        gradient = outputs.applyFunction(relu_deriv).mult_elements(outputGradient);
    } else {
        gradient = outputGradient;
    }

    Matrix inputsTransposed = inputs.transpose();
    Matrix currentWeightGradients = gradient.multiply(inputsTransposed);
    Matrix currentBiasGradients = gradient;
    
    weightGradientAccumulator = weightGradientAccumulator.add(currentWeightGradients);
    biasGradientAccumulator = biasGradientAccumulator.add(currentBiasGradients);
    
    return weights.transpose().multiply(gradient);

    // double grad_norm = 0.0;
    // for (int i = 0; i < gradient.get_rows(); i++) {
    //     grad_norm += gradient.get(i, 0) * gradient.get(i, 0);
    // }
    // grad_norm = std::sqrt(grad_norm);
    // std::cout << "Gradient norm: " << grad_norm << std::endl; // more debugging
}

void Layer::update_weights(double learning_rate, int batch_size) {
    // Matrix avgWeightGrad = weightGradientAccumulator.multiply_scalar(1.0 / batch_size);
    // Matrix avgBiasGrad = biasGradientAccumulator.multiply_scalar(1.0 / batch_size);
    
    // update
    weights = weights.subtract(weightGradientAccumulator.multiply_scalar(learning_rate));
    biases = biases.subtract(biasGradientAccumulator.multiply_scalar(learning_rate));
    
    // reset accumulators
    weightGradientAccumulator = Matrix(weights.get_rows(), weights.get_cols());
    biasGradientAccumulator = Matrix(biases.get_rows(), biases.get_cols());
}
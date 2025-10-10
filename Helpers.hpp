#pragma once
#include <iostream>
#include <cmath>

// helper functions from claude
// Activation functions (take pre-activation z)
inline double sigmoid(const double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

inline double relu(const double z) {
    return (z <= 0) ? 0 : z;
}

inline double tanh_activation(const double z) {
    return std::tanh(z);
}

// Derivatives (take pre-activation z)
inline double sigmoid_deriv(const double z) {
    double s = sigmoid(z);
    return s * (1 - s);
}

inline double relu_deriv(const double z) {
    return (z <= 0) ? 0 : 1;
}

inline double tanh_deriv(const double z) {
    double t = std::tanh(z);
    return 1 - t * t;
}

// Loss functions
inline double mse_loss(const double predicted, const double target) {
    double diff = predicted - target;
    return diff * diff;
}

inline double mse_loss_deriv(const double predicted, const double target) {
    return 2 * (predicted - target);
}

// Could also add cross-entropy for classification:
inline double binary_cross_entropy(const double predicted, const double target) {
    const double epsilon = 1e-7; // prevent log(0)
    return -(target * std::log(predicted + epsilon) + 
             (1 - target) * std::log(1 - predicted + epsilon));
}

inline double binary_cross_entropy_deriv(const double predicted, const double target) {
    const double epsilon = 1e-7;
    return -(target / (predicted + epsilon)) + 
           ((1 - target) / (1 - predicted + epsilon));
}
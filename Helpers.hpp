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

inline Matrix softmax(const Matrix& m) {
    Matrix result(m.get_rows(), m.get_cols());
    double sum = 0.0;
    
    // Find max for numerical stability
    double max_val = m.get(0, 0);
    for (int i = 1; i < m.get_rows(); i++) {
        if (m.get(i, 0) > max_val) max_val = m.get(i, 0);
    }
    
    // Compute exp and sum
    for (int i = 0; i < m.get_rows(); i++) {
        result.set(i, 0, std::exp(m.get(i, 0) - max_val));
        sum += result.get(i, 0);
    }
    
    // Normalize
    for (int i = 0; i < m.get_rows(); i++) {
        result.set(i, 0, result.get(i, 0) / sum);
    }
    
    return result;
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
#include "NeuralNet.hpp"
#include "Helpers.hpp"
#include "Matrix.hpp"
#include <chrono> // updates for data progress every second
#include <thread>

NeuralNetwork::NeuralNetwork(double lr) {
    learning_rate = lr;
}

void NeuralNetwork::addLayer(int inputSize, int outputSize) {
    layers.push_back(Layer(inputSize, outputSize));
}

Matrix NeuralNetwork::forward(Matrix& input) {
    Matrix output = input;

    for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i].forward(output);
    }

    return output;
}

void NeuralNetwork::backward(Matrix& target, Matrix& output) {
    Matrix gradient(output.get_rows(), output.get_cols());

    for (int i = 0; i < output.get_rows(); i++) {
        for (int j = 0; j < output.get_cols(); j++) {
            gradient.set(i, j, mse_loss_deriv(output.get(i, j), target.get(i, j)));
        }
    }

    for (int i = layers.size() - 1; i >= 0; i--) {
        gradient = layers[i].backward(gradient, learning_rate);
    }
}

double NeuralNetwork::calculateLoss(Matrix& predicted, Matrix& target) {
    double curr_loss = 0; 

    for (int i = 0; i < predicted.get_rows(); i++) {
        for (int j = 0; j < predicted.get_cols(); j++) {
            curr_loss += mse_loss(predicted.get(i, j), target.get(i, j));
        }
    }

    return curr_loss;
}

void NeuralNetwork::train(std::vector<Matrix>& data, std::vector<Matrix>& labels, int epochs) {
    size_t data_count = data.size();
    double total_loss = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto last_print = std::chrono::steady_clock::now();

        for (size_t i = 0; i < data.size(); i++) {
            Matrix output = forward(data[i]);

            total_loss += calculateLoss(output, labels[i]);
            backward(labels[i], output);

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_print);
            if (elapsed.count() >= 1) {
                std::cout << "Progress: " << i + 1 << "/" << data_count << std::endl;
                last_print = now;
            }
        }

        std::cout << "~~~~~~~~~~ EPOCH " << epoch << "finished! ~~~~~~~~~~" << std::endl;
    }

    std::cout << "total loss: " << total_loss;
}
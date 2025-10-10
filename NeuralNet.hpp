#pragma once
#include <iostream>
#include <vector>
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Helpers.hpp"


class NeuralNetwork {
private:
    std::vector<Layer> layers;
    double learning_rate;

public:
    NeuralNetwork(double lr);
    void addLayer(int inputSize, int outputSize);
    Matrix forward(Matrix& input);
    void backward(Matrix& target, Matrix& output);
    double calculateLoss(Matrix& predicted, Matrix& target);
    void train(std::vector<Matrix>& data, std::vector<Matrix>& labels, int epochs);
};
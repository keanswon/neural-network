#include "NeuralNet.hpp"
#include "Helpers.hpp"
#include "Matrix.hpp"
#include <chrono> // updates for data progress every second
#include <thread>
#include <fstream>

NeuralNetwork::NeuralNetwork(double lr) {
    learning_rate = lr;
}

void NeuralNetwork::addLayer(int inputSize, int outputSize, Activation act) {
    layers.push_back(Layer(inputSize, outputSize, act));
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
            gradient.set(i, j, output.get(i, j) - target.get(i, j));
        }
    }

    for (int i = layers.size() - 1; i >= 0; i--) {
        gradient = layers[i].backward(gradient, learning_rate);
    }
}

double NeuralNetwork::calculateLoss(Matrix& predicted, Matrix& target) {
    double total_loss = 0.0;
    const double epsilon = 1e-7;
    
    for (int i = 0; i < predicted.get_rows(); i++) {
        if (target.get(i, 0) > 0.5) {  // This is the true class
            total_loss -= std::log(predicted.get(i, 0) + epsilon);
        }
    }
    return total_loss;
}

void NeuralNetwork::train(std::vector<Matrix>& data, std::vector<Matrix>& labels, int epochs) {
    size_t data_count = data.size();
    double total_loss = 0.0;
    auto total_timer = std::chrono::steady_clock::now(); // timer for total time

    std::random_device rd; // for shuffling data
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto last_print = std::chrono::steady_clock::now(); // timer to print every second
        auto epoch_timer = std::chrono::steady_clock::now(); // timer to time the current epoch

        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < indices.size(); i++) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t i = 0; i < data.size(); i++) {
            size_t idx = indices[i]; // use shuffled indices for randomization between training data sets
            Matrix output = forward(data[idx]);

            total_loss += calculateLoss(output, labels[idx]);
            backward(labels[idx], output);

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_print);
            if (elapsed.count() >= 1) {
                std::cout << "Progress: " << i + 1 << "/" << data_count << std::endl;
                last_print = now;
            }
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_timer);
        
        int total_seconds = epoch_duration.count();
        int minutes = total_seconds / 60;
        int seconds = total_seconds % 60;
        
        std::cout << "~~~~ EPOCH " << epoch + 1 << " finished in " 
                << minutes << "m " << seconds << "s ~~~~" << std::endl;

        learning_rate *= 0.85; // decay learning rate every epoch
    }

    auto timer_end = std::chrono::steady_clock::now();
    auto timer_duration = std::chrono::duration_cast<std::chrono::seconds>(timer_end - total_timer);

    int total_seconds = timer_duration.count();
    int minutes = total_seconds / 60;
    int seconds = total_seconds % 60;

    std::cout << "total loss: " << total_loss << std::endl;
    std::cout << "total time taken: " << minutes << "m " << seconds << "s" << std::endl;
}


/** 
 * 
 * writes model in the form:
 * 
 * [ n layers ]
 * [ for each layer: ]
 *  [ number of weight rows ]
 *  [ number of weight cols ]
 *  [ weight data ]
 *  [ number of bias rows ]
 *  [ number of bias cols ]
 *  [ bias data ]
 * 
 * does not save the architecture of the network, that's a problem for a future version lol
 * 
 * saving the architecture of the network would allow easier loading of different weights / biases,
 * more of a modular approach
 */


void NeuralNetwork::save_model(std::string filepath) {
    std::ofstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Cannot open file for saving: " << filepath << std::endl;
        return;
    }

    int num_layers = layers.size();
    file.write((char*)&num_layers, sizeof(int));

    for (int i = 0; i < num_layers; i++) {
        int act_type = static_cast<int>(layers[i].activation_type);
        file.write((char*)&act_type, sizeof(int));

        int weight_rows = layers[i].weights.get_rows();
        int weight_cols = layers[i].weights.get_cols();

        file.write((char*)&weight_rows, sizeof(int));
        file.write((char*)&weight_cols, sizeof(int));

        for (int r = 0; r < weight_rows; r++) {
            for (int c = 0; c < weight_cols; c++) {
                double val = layers[i].weights.get(r, c);
                file.write((char*)&val, sizeof(double));
            }
        }

        int bias_rows = layers[i].biases.get_rows();
        int bias_cols = layers[i].biases.get_cols();

        file.write((char*)&bias_rows, sizeof(int));
        file.write((char*)&bias_cols, sizeof(int));

        for (int r = 0; r < bias_rows; r++) {
            for (int c = 0; c < bias_cols; c++) {
                double bias = layers[i].biases.get(r, c);
                file.write((char*)&bias, sizeof(double));
            }
        }
    }

    file.close();
    std::cout << "Model saved to " << filepath << std::endl;
}

void NeuralNetwork::load_model(std::string filepath) {
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Cannot open file for loading: " << filepath << std::endl;
        return;
    }

    layers.clear();

    int num_layers;
    file.read((char*)&num_layers, sizeof(int));

    for (int i = 0; i < num_layers; i++) {
        int act_type;
        file.read((char*)&act_type, sizeof(int));
        Activation activation = static_cast<Activation>(act_type);

        int weight_rows, weight_cols;
        file.read((char*)&weight_rows, sizeof(int));
        file.read((char*)&weight_cols, sizeof(int));

        layers.push_back(Layer(weight_cols, weight_rows, activation));

        for (int r = 0; r < weight_rows; r++) {
            for (int c = 0; c < weight_cols; c++) {
                double val;
                file.read((char*)&val, sizeof(double));
                layers[i].weights.set(r, c, val);
            }
        }

        int bias_rows, bias_cols;
        file.read((char*)&bias_rows, sizeof(int));
        file.read((char*)&bias_cols, sizeof(int));

        for (int r = 0; r < bias_rows; r++) {
            for (int c = 0; c < bias_cols; c++) {
                double val;
                file.read((char*)&val, sizeof(double));
                layers[i].biases.set(r, c, val);
            }
        }
    }

    file.close();
    std::cout << "Model loaded from " << filepath << std::endl;
}
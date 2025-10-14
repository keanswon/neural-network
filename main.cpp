#include "NeuralNet.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

const std::string MODEL_PATH = "models/model2.bin";
const std::string TRAIN_IMAGES = "MNIST/train-images-idx3-ubyte";
const std::string TRAIN_LABELS = "MNIST/train-labels-idx1-ubyte";
const std::string TEST_IMAGES = "MNIST/t10k-images-idx3-ubyte";
const std::string TEST_LABELS = "MNIST/t10k-labels-idx1-ubyte";
double learning_rate = .0005;
double decay_rate = 0.9;
int batch_size = 32;

std::unordered_map<std::string, std::vector<std::string>> filepath_to_model_iteration; // given a filepath, stores information about it

int reverseInt(int i);
Matrix labelToOneHot(unsigned char label);
void train_model(const std::string& images_file, const std::string& labels_file, int num_epochs);
void test_model(const std::string& images_file, const std::string& labels_file);
static int argmax(const Matrix& m);

int main() {
    train_model(TRAIN_IMAGES, TRAIN_LABELS, 10);
    test_model(TEST_IMAGES, TEST_LABELS);
    
    return 0;
}

// helpers for reading file
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Matrix labelToOneHot(unsigned char label) {
    Matrix oneHot(10, 1);  // 10 rows (digits 0-9), 1 column
    oneHot.set(label, 0, 1.0);
    return oneHot;
}

static int argmax(const Matrix& m) {
    int maxIdx = 0;
    double maxVal = m.get(0, 0);
    
    for (int i = 1; i < m.get_rows(); i++) {
        if (m.get(i, 0) > maxVal) {
            maxVal = m.get(i, 0);
            maxIdx = i;
        }
    }
    return maxIdx;
}

// actually train the model
void train_model(const std::string& image_filepath, const std::string& label_filepath, int num_epochs) {
    std::ifstream images_file(image_filepath, std::ios::binary);
    if (!images_file.is_open()) {
        std::cerr << "Cannot open image file!" << std::endl;
    }
    
    // Open the label file
    std::ifstream labels_file(label_filepath, std::ios::binary);
    if (!labels_file.is_open()) {
        std::cerr << "Cannot open label file!" << std::endl;
    }
    // Read image file header
    int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
    
    images_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    
    images_file.read((char*)&n_images, sizeof(n_images));
    n_images = reverseInt(n_images);
    
    images_file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    
    images_file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    
    // std::cout << "Images: " << n_images << std::endl;
    // std::cout << "Size: " << n_rows << "x" << n_cols << std::endl;
    
    // Read label file header
    int magic_number_labels = 0, n_labels = 0;
    
    labels_file.read((char*)&magic_number_labels, sizeof(magic_number_labels));
    magic_number_labels = reverseInt(magic_number_labels);
    
    labels_file.read((char*)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);
    
    // std::cout << "Labels: " << n_labels << std::endl;
    

    std::vector<Matrix> images;

    images.reserve(n_images);

    unsigned char buffer[784];

    for (int img = 0; img < n_images; img++) {
        images_file.read((char*)buffer, 784);
        
        // make a column vector, this is 62x faster than reading byte by byte
        images.emplace_back(784, 1, buffer, 1.0/255.0);
    }

    // std::cout << images.size() << " images loaded" << std::endl;
    images_file.close();
    
    // Read one label
    std::vector<Matrix> labels;
    labels.reserve(n_images);

    for (int i = 0; i < n_images; i++) {  // Changed variable name to avoid shadowing
        unsigned char label = 0;
        labels_file.read((char*)&label, 1);
        Matrix label_one_hot = labelToOneHot(label);
        // std::cout << "label: " << argmax(label_one_hot) << std::endl;
        labels.emplace_back(label_one_hot);  // Use the one-hot conversion
    }

    // std::cout << labels.size() << " labels loaded" << std::endl;
    labels_file.close();
    
    NeuralNetwork number_gooner(learning_rate);
    number_gooner.addLayer(784, 256, Activation::RELU);
    number_gooner.addLayer(256, 128, Activation::RELU);
    number_gooner.addLayer(128, 64, Activation::RELU);
    number_gooner.addLayer(64, 10, Activation::SOFTMAX);

    number_gooner.train(images, labels, num_epochs, batch_size, decay_rate);

    number_gooner.save_model(MODEL_PATH);
}

// test the model
void test_model(const std::string& image_filepath, const std::string& label_filepath) {
    std::ifstream images_file(image_filepath, std::ios::binary);
    if (!images_file.is_open()) {
        std::cerr << "Cannot open image file!" << std::endl;
    }
    
    // Open the label file
    std::ifstream labels_file(label_filepath, std::ios::binary);
    if (!labels_file.is_open()) {
        std::cerr << "Cannot open label file!" << std::endl;
    }

    // reading is the same as training the model
    int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
    
    images_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    
    images_file.read((char*)&n_images, sizeof(n_images));
    n_images = reverseInt(n_images);
    
    images_file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    
    images_file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    
    // std::cout << "Images: " << n_images << std::endl;
    // std::cout << "Size: " << n_rows << "x" << n_cols << std::endl;
    
    // Read label file header
    int magic_number_labels = 0, n_labels = 0;
    
    labels_file.read((char*)&magic_number_labels, sizeof(magic_number_labels));
    magic_number_labels = reverseInt(magic_number_labels);
    
    labels_file.read((char*)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);
    
    // std::cout << "Labels: " << n_labels << std::endl;

    std::vector<Matrix> images;

    images.reserve(n_images);

    unsigned char buffer[784];

    for (int img = 0; img < n_images; img++) {
        images_file.read((char*)buffer, 784);
        
        // make a column vector, this is 62x faster than reading byte by byte
        images.emplace_back(784, 1, buffer, 1.0/255.0);
    }

    std::cout << images.size() << " images loaded" << std::endl;
    images_file.close();
    
    // read one label
    std::vector<Matrix> labels;
    labels.reserve(n_images);

    for (int i = 0; i < n_images; i++) {  // Changed variable name to avoid shadowing
        unsigned char label = 0;
        labels_file.read((char*)&label, 1);
        labels.emplace_back(labelToOneHot(label));  // Use the one-hot conversion
    }

    std::cout << labels.size() << " labels loaded" << std::endl;
    labels_file.close();

    NeuralNetwork number_gooner = NeuralNetwork(learning_rate);
    number_gooner.load_model(MODEL_PATH);

    int num_incorrect = 0;
    int total_images = n_images;

    for (int i = 0; i < n_images; i++) {
        int result = argmax(number_gooner.forward(images[i]));
        // std::cout << "Predicted: " << result << ", Actual: " << argmax(labels[i]) << std::endl;

        int curr_label = argmax(labels[i]);

        if (result != curr_label) num_incorrect++;
    }

    int num_correct = total_images - num_incorrect;
    double accuracy = static_cast<double>(num_correct) / total_images;
    std::cout << "accuracy: " << num_correct << "/" << total_images
            << " (" << accuracy * 100.0 << "%)\n";

}
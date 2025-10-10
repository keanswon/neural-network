#include "NeuralNet.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

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

int main() {
    // main from claude to open / convert files to arrays
    // Open the image file
    std::ifstream imageFile("MNIST/train-images-idx3-ubyte", std::ios::binary);
    if (!imageFile.is_open()) {
        std::cerr << "Cannot open image file!" << std::endl;
        return 1;
    }
    
    // Open the label file
    std::ifstream labelFile("MNIST/train-labels-idx1-ubyte", std::ios::binary);
    if (!labelFile.is_open()) {
        std::cerr << "Cannot open label file!" << std::endl;
        return 1;
    }
    
    // Read image file header
    int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
    
    imageFile.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    
    imageFile.read((char*)&n_images, sizeof(n_images));
    n_images = reverseInt(n_images);
    
    imageFile.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    
    imageFile.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    
    std::cout << "Images: " << n_images << std::endl;
    std::cout << "Size: " << n_rows << "x" << n_cols << std::endl;
    
    // Read label file header
    int magic_number_labels = 0, n_labels = 0;
    
    labelFile.read((char*)&magic_number_labels, sizeof(magic_number_labels));
    magic_number_labels = reverseInt(magic_number_labels);
    
    labelFile.read((char*)&n_labels, sizeof(n_labels));
    n_labels = reverseInt(n_labels);
    
    std::cout << "Labels: " << n_labels << std::endl;
    

    std::vector<Matrix> images;
    size_t num_images = 60000;

    // use 60,000 images for now
    images.reserve(num_images);

    unsigned char buffer[784];

    for (size_t img = 0; img < num_images; img++) {
        imageFile.read((char*)buffer, 784);
        
        // make a column vector, this is 62x faster than reading byte by byte
        images.emplace_back(784, 1, buffer, 1.0/255.0);
    }

    std::cout << images.size() << "images loaded" << std::endl;
    
    // Read one label
    std::vector<Matrix> labels;
    labels.reserve(num_images);

    for (size_t i = 0; i < num_images; i++) {  // Changed variable name to avoid shadowing
        unsigned char label = 0;
        labelFile.read((char*)&label, 1);
        labels.emplace_back(labelToOneHot(label));  // Use the one-hot conversion
    }

    std::cout << labels.size() << " labels loaded" << std::endl;
    
    NeuralNetwork number_gooner(.001);
    number_gooner.addLayer(784, 256);
    number_gooner.addLayer(256, 128);
    number_gooner.addLayer(128, 10);

    number_gooner.train(images, labels, 5);
    
    imageFile.close();
    labelFile.close();
    
    return 0;
}
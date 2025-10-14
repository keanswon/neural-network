# MNIST Neural Network (C++)

A fully-connected neural network implementation from scratch in C++ for digit recognition on the MNIST dataset.

## Features

- **Pure C++ Implementation**: No external ML libraries (TensorFlow, PyTorch, etc.)
- **Custom Matrix Operations**: Matrix class with all necessary operations
- **Flexible Architecture**: Easy to configure layer sizes and activation functions
- **Multiple Activation Functions**: ReLU, Softmax, Sigmoid, Tanh
- **Model Persistence**: Save and load trained models
- **Batch Training**: Configurable batch size for gradient descent
- **Learning Rate Decay**: Automatic learning rate reduction per epoch

## Architecture

The current network configuration:
- **Input Layer**: 784 neurons (28×28 flattened images)
- **Hidden Layer 1**: 256 neurons (ReLU activation)
- **Hidden Layer 2**: 128 neurons (ReLU activation)
- **Hidden Layer 3**: 64 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax activation)

## Project Structure

```
├── main.cpp          # Entry point, data loading, training/testing
├── NeuralNet.cpp/hpp # Neural network class
├── Layer.cpp/hpp     # Individual layer implementation
├── Matrix.hpp        # Matrix operations (header-only)
├── Helpers.hpp       # Activation functions and utilities
└── models/           # Saved model directory
```

## Requirements

- C++11 or later
- Standard C++ libraries only (no external dependencies)
- MNIST dataset files (see below)

## Dataset Setup

Download the MNIST dataset and place the files in the following structure:

```
MNIST/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

## Compilation

make

### Training

```cpp
// Configure in main.cpp
const double learning_rate = 0.0005;
const int num_epochs = 3;

// Run training
train_model(TRAIN_IMAGES, TRAIN_LABELS, num_epochs);
```

### Testing

```cpp
// Test the trained model
test_model(TEST_IMAGES, TEST_LABELS);
```

### Running the Program

```bash
./neuralnet
```

The program will:
1. Train the network on 60,000 training images
2. Save the model to `models/model1.bin`
3. Test on 10,000 test images
4. Display accuracy metrics

## Key Features

### Matrix Class
- Efficient matrix multiplication, addition, subtraction
- Element-wise operations (Hadamard product)
- Transpose operation
- Function application to all elements
- Move semantics for performance

### Training Features
- **Mini-batch Gradient Descent**: Configurable batch size (default: 32)
- **Data Shuffling**: Random shuffling each epoch for better convergence
- **Progress Tracking**: Real-time progress updates during training
- **Learning Rate Decay**: 0.85× decay factor per epoch
- **Cross-Entropy Loss**: For multi-class classification

### Model Saving/Loading
Models are saved in binary format containing:
- Number of layers
- Activation types
- Weight matrices
- Bias vectors

## Performance

Training time depends on hardware, but typical performance:
- **~2-5 minutes per epoch** on modern CPUs
- **~92-95% accuracy** on test set after 3 epochs

## Customization

### Changing Network Architecture

```cpp
NeuralNetwork nn(learning_rate);
nn.addLayer(784, 512, Activation::RELU);
nn.addLayer(512, 256, Activation::RELU);
nn.addLayer(256, 10, Activation::SOFTMAX);
```

### Adding New Activation Functions

1. Add function to `Helpers.hpp`:
```cpp
inline double my_activation(const double z) {
    // your implementation
}

inline double my_activation_deriv(const double z) {
    // derivative
}
```

2. Add to `Activation` enum in `Layer.hpp`
3. Update `Layer::forward()` and `Layer::backward()`

### Adjusting Hyperparameters

```cpp
// Learning rate
const double learning_rate = 0.001;

// Batch size
nn.train(images, labels, epochs, 64);

// Learning rate decay (in NeuralNet.cpp)
learning_rate *= 0.90;  // Change decay factor
```

## Known Limitations

- No GPU acceleration
- Sequential processing only (no parallelization)
- Fixed architecture must match during loading
- No regularization (dropout, L2) implemented
- No advanced optimizers (Adam, RMSprop)

## Future Improvements

- [ ] Implement Adam optimizer
- [ ] Add dropout regularization
- [ ] Multi-threading for matrix operations
- [ ] Convolutional layers
- [ ] More flexible model save format
- [ ] Validation set evaluation during training
- [ ] Early stopping

## License

This is an educational project. Feel free to use and modify as needed.

## Acknowledgments

- MNIST dataset by Yann LeCun
- Math from 3Blue1Brown
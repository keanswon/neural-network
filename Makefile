# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -funroll-loops
LDFLAGS = 

# Target executable
TARGET = neuralnet

# Source files
SRCS = main.cpp Layer.cpp NeuralNet.cpp

# Object files (replace .cpp with .o)
OBJS = $(SRCS:.cpp=.o)

# Header files (for dependency tracking)
HEADERS = Matrix.hpp Layer.hpp NeuralNet.hpp Helpers.h

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Rebuild everything from scratch
rebuild: clean all

# Phony targets (not actual files)
.PHONY: all clean run rebuild
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> // For std::exception
#include <iomanip>   // For std::fixed and std::setprecision

#include "tensor.h"
#include "ml_layers.h" // This should bring in Layer, ConvolutionLayer, MaxPoolingLayer, AveragePoolingLayer

// Helper function to print tensor summary
void print_tensor_summary(const Tensor& t, const std::string& name) {
    std::cout << "--- Tensor Summary: " << name << " ---" << std::endl;
    std::cout << "Shape: [N=" << t.getN() 
              << ", C=" << t.getC()
              << ", H=" << t.getH()
              << ", W=" << t.getW() << "]" << std::endl;
    std::cout << "Total elements: " << t.getSize() << std::endl;
    
    if (t.getData() == nullptr && t.getSize() > 0) {
        std::cout << "Data pointer is null!" << std::endl;
        return;
    }
    if (t.getSize() == 0) {
        std::cout << "Tensor is empty." << std::endl;
        return;
    }

    std::cout << "First few values (up to 10 or actual size):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    size_t print_count = std::min(static_cast<size_t>(10), t.getSize());
    const float* data = t.getData();
    for (size_t i = 0; i < print_count; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    if (t.getSize() > print_count) {
        std::cout << "... (omitting remaining " << t.getSize() - print_count << " values)";
    }
    std::cout << std::endl << "------------------------------------" << std::endl;
}

int main() {
    std::cout << "Starting ML Layer Demonstration..." << std::endl;

    // 1. Create Sample Input Tensor
    int N = 1, C_in = 1, H_in = 8, W_in = 8;
    Tensor input_tensor(N, C_in, H_in, W_in);
    float* input_data = input_tensor.getData();
    for (size_t i = 0; i < input_tensor.getSize(); ++i) {
        input_data[i] = static_cast<float>(i % 10); // Simple pattern 0-9 repeating
    }
    print_tensor_summary(input_tensor, "Input Tensor");

    // 2. Instantiate Layers
    // Convolution Layer: 1 input channel, 2 output filters, 3x3 kernel, stride 1, padding 1
    // Output H/W = (8 - 3 + 2*1)/1 + 1 = 8. Output shape: [1, 2, 8, 8]
    ConvolutionLayer conv_layer(input_tensor.getC(), /*num_filters=*/2, /*kernel_size=*/3, /*stride=*/1, /*padding=*/1, "Conv1");
    
    // Max Pooling Layer: 2x2 pool, stride 2
    // Input from conv_output (1,2,8,8). Output H/W = (8-2)/2 + 1 = 4. Output shape: [1, 2, 4, 4]
    MaxPoolingLayer max_pool_layer(/*pool_size=*/2, /*stride=*/2, "MaxPool1");

    // Average Pooling Layer: 2x2 pool, stride 2
    // Input from max_pool_output (1,2,4,4). Output H/W = (4-2)/2+1 = 2. Output shape: [1,2,2,2]
    AveragePoolingLayer avg_pool_layer(/*pool_size=*/2, /*stride=*/2, "AvgPool1");

    Tensor conv_output(0,0,0,0); // Default constructor or placeholder
    Tensor max_pool_output(0,0,0,0);
    Tensor avg_pool_output(0,0,0,0);

    // 3. Demonstrate Forward Passes Sequentially
    try {
        std::cout << "\n--- Applying " << conv_layer.getName() << " ---" << std::endl;
        conv_output = conv_layer.forward(input_tensor);
        print_tensor_summary(conv_output, conv_layer.getName() + " Output");

        std::cout << "\n--- Applying " << max_pool_layer.getName() << " to " << conv_layer.getName() << " output ---" << std::endl;
        max_pool_output = max_pool_layer.forward(conv_output);
        print_tensor_summary(max_pool_output, max_pool_layer.getName() + " Output");
        
        std::cout << "\n--- Applying " << avg_pool_layer.getName() << " to " << max_pool_layer.getName() << " output ---" << std::endl;
        avg_pool_output = avg_pool_layer.forward(max_pool_output);
        print_tensor_summary(avg_pool_output, avg_pool_layer.getName() + " Output");

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nML Layer Demonstration Complete." << std::endl;

    return 0;
}

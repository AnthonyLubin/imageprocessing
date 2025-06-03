#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> // For std::exception
#include <iomanip>   // For std::fixed and std::setprecision

#include "ImageML_Lib/tensor.h"
#include "ImageML_Lib/ml_layers.h" // This should bring in Layer, ConvolutionLayer, MaxPoolingLayer, AveragePoolingLayer

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
    std::cout << "------------------------------------\n" << std::endl;
}

void demonstrate_dense_relu_flatten_pipeline() {
    std::cout << "Starting Dense/ReLU/Flatten Pipeline Demonstration..." << std::endl;

    // 1. Sample Input Tensor (e.g., output from a pooling layer)
    // N=1, C=2, H=2, W=2. Total features = 2*2*2 = 8
    Tensor input_features(1, 2, 2, 2);
    float* input_data_mlp = input_features.getData();
    for (size_t i = 0; i < input_features.getSize(); ++i) {
        input_data_mlp[i] = static_cast<float>(i + 1) * 0.5f; // Small positive values
    }
    print_tensor_summary(input_features, "Initial Input for MLP Pipeline");

    // 2. Layer Instantiation
    ReLULayer relu1_mlp("ReLU1_MLP");
    FlattenLayer flatten_layer_mlp("Flatten_MLP");

    int flattened_size = input_features.getC() * input_features.getH() * input_features.getW();
    if (flattened_size == 0 && input_features.getN() > 0) { // Should not happen with N>0 and C,H,W >0
        std::cerr << "Warning: Calculated flattened size is 0 for non-empty batch." << std::endl;
        // Potentially throw or handle as an error state for the demo
    }
    // Ensure flattened_size is at least 1 if input N is also 0 (empty tensor)
    // For this demo, input_features is (1,2,2,2) so flattened_size will be 8.

    DenseLayer dense_layer_mlp(flattened_size, 5, "Dense1_MLP"); // 8 input features, 5 output features
    ReLULayer relu2_mlp("ReLU2_MLP");

    Tensor relu1_out_mlp(0,0,0,0), flatten_out_mlp(0,0,0,0), dense_out_mlp(0,0,0,0), final_out_mlp(0,0,0,0);

    try {
        // 3. Forward Pass
        std::cout << "\n--- Starting MLP Forward Pass ---" << std::endl;
        relu1_out_mlp = relu1_mlp.forward(input_features);
        print_tensor_summary(relu1_out_mlp, "After ReLU1_MLP");

        flatten_out_mlp = flatten_layer_mlp.forward(relu1_out_mlp);
        print_tensor_summary(flatten_out_mlp, "After Flatten_MLP");
        // Expected shape: (1, 8, 1, 1)

        dense_out_mlp = dense_layer_mlp.forward(flatten_out_mlp);
        print_tensor_summary(dense_out_mlp, "After Dense1_MLP");
        // Expected shape: (1, 5, 1, 1)

        final_out_mlp = relu2_mlp.forward(dense_out_mlp);
        print_tensor_summary(final_out_mlp, "Final Output (After ReLU2_MLP)");

        // 4. Conceptual Backward Pass
        std::cout << "\n--- Starting Conceptual MLP Backward Pass ---" << std::endl;

        // Create a dummy gradient for the final output (Tensor of ones with same shape as final_out_mlp)
        Tensor dummy_output_gradient(final_out_mlp.getN(), final_out_mlp.getC(), final_out_mlp.getH(), final_out_mlp.getW());
        float* grad_data = dummy_output_gradient.getData();
        if (grad_data) {
            for (size_t i = 0; i < dummy_output_gradient.getSize(); ++i) {
                grad_data[i] = 1.0f;
            }
        } else if (dummy_output_gradient.getSize() > 0) {
             throw std::runtime_error("Failed to get data pointer for dummy_output_gradient despite non-zero size.");
        }
        print_tensor_summary(dummy_output_gradient, "Initial Output Gradient (Dummy Ones)");

        Tensor grad_from_relu2 = relu2_mlp.backward(dummy_output_gradient);
        print_tensor_summary(grad_from_relu2, "Grad from ReLU2_MLP");

        Tensor grad_from_dense = dense_layer_mlp.backward(grad_from_relu2);
        print_tensor_summary(grad_from_dense, "Grad from Dense1_MLP");

        Tensor grad_from_flatten = flatten_layer_mlp.backward(grad_from_dense);
        print_tensor_summary(grad_from_flatten, "Grad from Flatten_MLP");

        Tensor grad_from_relu1 = relu1_mlp.backward(grad_from_flatten);
        print_tensor_summary(grad_from_relu1, "Grad from ReLU1_MLP (Input Grad for MLP)");

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during MLP pipeline demonstration: " << e.what() << std::endl;
    }
    std::cout << "\nDense/ReLU/Flatten Pipeline Demonstration Complete." << std::endl;
    std::cout << "------------------------------------\n" << std::endl;
}


int main() {
    std::cout << "Starting ML Layer Demonstration..." << std::endl;

    // Convolutional/Pooling Pipeline Demo (Original Demo)
    // ... (existing try-catch block for conv_pool_pipeline) ...
    // For brevity, I'm assuming the original conv/pool demo is here.
    // If it was meant to be removed, this main function would only call the new demo.
    // To make this runnable, let's simulate its presence or comment it out.

    // Simulate original demo execution or comment it out if it's too verbose for this step
    // For now, let's assume the original demo code was here.
    // To avoid making this diff too large, I'll just call the previous demo function name
    // which was implicitly the content of main before.
    // This function does not exist, so in a real run, this would be the old code from main.
    // run_conv_pool_demo(); // Placeholder for the original main's content

    // To properly test, let's make main only run the new demo for now,
    // or ensure the old demo is also wrapped in a function.
    // For this specific task, let's assume we are just adding a new demo.
    // The previous demo content:
    // Tensor input_tensor(N, C_in, H_in, W_in); ... conv_layer ... etc.
    // This should be encapsulated if both demos are to run.

    // For this task, let's just run the new demo.
    // The original demo content from main() is assumed to be refactored if needed.
    // If the original demo was inside the try-catch block in main:
    try {
        // --- Convolutional Pipeline Demo (from previous main) ---
        int N_conv = 1, C_in_conv = 1, H_in_conv = 8, W_in_conv = 8;
        Tensor input_tensor_conv(N_conv, C_in_conv, H_in_conv, W_in_conv);
        float* input_data_conv = input_tensor_conv.getData();
        for (size_t i = 0; i < input_tensor_conv.getSize(); ++i) {
            input_data_conv[i] = static_cast<float>(i % 10);
        }
        print_tensor_summary(input_tensor_conv, "Input Tensor (Conv Pipeline)");

        ConvolutionLayer conv_layer(input_tensor_conv.getC(), 2, 3, 1, 1, "Conv1");
        MaxPoolingLayer max_pool_layer(2, 2, "MaxPool1");
        AveragePoolingLayer avg_pool_layer(2, 2, "AvgPool1");

        std::cout << "\n--- Applying " << conv_layer.getName() << " ---" << std::endl;
        Tensor conv_output = conv_layer.forward(input_tensor_conv);
        print_tensor_summary(conv_output, conv_layer.getName() + " Output");

        std::cout << "\n--- Applying " << max_pool_layer.getName() << " to " << conv_layer.getName() << " output ---" << std::endl;
        Tensor max_pool_output = max_pool_layer.forward(conv_output);
        print_tensor_summary(max_pool_output, max_pool_layer.getName() + " Output");

        std::cout << "\n--- Applying " << avg_pool_layer.getName() << " to " << max_pool_layer.getName() << " output ---" << std::endl;
        Tensor avg_pool_output = avg_pool_layer.forward(max_pool_output);
        print_tensor_summary(avg_pool_output, avg_pool_layer.getName() + " Output");
        std::cout << "\nConvolutional/Pooling Pipeline Demonstration Complete." << std::endl;
        std::cout << "------------------------------------\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during Conv/Pool pipeline: " << e.what() << std::endl;
    }


    // --- New Dense/ReLU/Flatten Pipeline Demo ---
    demonstrate_dense_relu_flatten_pipeline();

    return 0;
}

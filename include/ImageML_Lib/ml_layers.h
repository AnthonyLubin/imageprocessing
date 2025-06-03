#ifndef ML_LAYERS_H
#define ML_LAYERS_H

#include "ImageML_Lib/tensor.h" // For Tensor class
#include <string>    // For std::string
#include <utility>   // For std::move

// Abstract base class for all neural network layers
class Layer {
public:
    // Constructor that takes a layer name
    Layer(std::string name) : layer_name_(std::move(name)) {}

    // Virtual destructor to ensure proper cleanup of derived classes
    virtual ~Layer() = default;

    // Pure virtual forward method: defines the forward pass computation
    // Takes a const reference to an input Tensor and returns an output Tensor.
    virtual Tensor forward(const Tensor& input) = 0;

    // Pure virtual backward method: defines the backward pass computation (for backpropagation)
    // Takes a const reference to the gradient of the loss with respect to the layer's output,
    // and returns the gradient of the loss with respect to the layer's input.
    virtual Tensor backward(const Tensor& output_gradient) = 0;

    // Getter for the layer's name (useful for debugging/logging)
    const std::string& getName() const {
        return layer_name_;
    }

protected:
    std::string layer_name_;
};


// --- ConvolutionLayer ---
class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer(int input_channels, int num_filters, int kernel_size, 
                     int stride = 1, int padding = 0, std::string name = "ConvolutionLayer")
        : Layer(std::move(name)),
          input_channels_(input_channels),
          num_filters_(num_filters),
          kernel_size_(kernel_size),
          stride_(stride),
          padding_(padding),
          weights_(num_filters, input_channels, kernel_size, kernel_size),
          biases_(1, num_filters, 1, 1) // N=1, C=num_filters, H=1, W=1 for easy broadcasting
    {
        if (input_channels <= 0 || num_filters <= 0 || kernel_size <= 0 || stride <= 0 || padding < 0) {
            throw std::invalid_argument("ConvolutionLayer parameters are invalid.");
        }
        // Initialize weights with small random values
        float* w_data = weights_.getData();
        size_t w_size = weights_.getSize();
        for (size_t i = 0; i < w_size; ++i) {
            w_data[i] = ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * 0.1f; // Small random values ~[-0.05, 0.05]
        }
        // Biases are already initialized to zero by Tensor constructor
    }

    Tensor forward(const Tensor& input) override {
        // Input Validation
        if (input.getN() == 0 || input.getC() == 0 || input.getH() == 0 || input.getW() == 0) {
            throw std::invalid_argument("ConvolutionLayer: Input tensor dimensions cannot be zero.");
        }
        if (input.getC() != input_channels_) {
            throw std::invalid_argument("ConvolutionLayer: Input channels mismatch. Expected " + 
                                        std::to_string(input_channels_) + ", got " + std::to_string(input.getC()));
        }
        if (input.getN() < 1 || input.getC() < 1 || input.getH() < 1 || input.getW() < 1) { // Redundant due to Tensor constructor, but good check
             throw std::invalid_argument("ConvolutionLayer: Input tensor dimensions must be positive.");
        }


        // Output Dimensions Calculation
        int N = input.getN();
        int input_H = input.getH();
        int input_W = input.getW();

        int output_H = (input_H - kernel_size_ + 2 * padding_) / stride_ + 1;
        int output_W = (input_W - kernel_size_ + 2 * padding_) / stride_ + 1;

        if (output_H <= 0 || output_W <= 0) {
            throw std::invalid_argument("ConvolutionLayer: Calculated output dimensions are non-positive. "
                                        "Check kernel_size, stride, padding, and input dimensions. "
                                        "Output H: " + std::to_string(output_H) + 
                                        ", Output W: " + std::to_string(output_W));
        }

        // Output Tensor Creation
        Tensor output(N, num_filters_, output_H, output_W);

        // Convolution Operation
        for (int n_idx = 0; n_idx < N; ++n_idx) { // Batch
            for (int f_idx = 0; f_idx < num_filters_; ++f_idx) { // Filters
                for (int oh_idx = 0; oh_idx < output_H; ++oh_idx) { // Output Height
                    for (int ow_idx = 0; ow_idx < output_W; ++ow_idx) { // Output Width
                        
                        float sum = 0.0f;
                        int ih_start = oh_idx * stride_ - padding_;
                        int iw_start = ow_idx * stride_ - padding_;

                        for (int ic_idx = 0; ic_idx < input_channels_; ++ic_idx) { // Input Channels
                            for (int kh_idx = 0; kh_idx < kernel_size_; ++kh_idx) { // Kernel Height
                                for (int kw_idx = 0; kw_idx < kernel_size_; ++kw_idx) { // Kernel Width
                                    
                                    int current_ih = ih_start + kh_idx;
                                    int current_iw = iw_start + kw_idx;

                                    // Check bounds (for padding)
                                    if (current_ih >= 0 && current_ih < input_H &&
                                        current_iw >= 0 && current_iw < input_W) {
                                        sum += input.at(n_idx, ic_idx, current_ih, current_iw) *
                                               weights_.at(f_idx, ic_idx, kh_idx, kw_idx);
                                    }
                                }
                            }
                        }
                        // Add bias
                        output.at(n_idx, f_idx, oh_idx, ow_idx) = sum + biases_.at(0, f_idx, 0, 0);
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        // Placeholder implementation
        // A real implementation would calculate gradients with respect to input (dX)
        // and with respect to weights (dW) and biases (dB).
        // For dX, it would typically involve a "transposed" or "full" convolution.
        // For now, we just acknowledge the call and indicate it's not implemented.

        // To return a tensor of the correct input gradient shape, we'd need the original input shape.
        // This is often stored during the forward pass in more complex frameworks.
        // For simplicity here, we'll just throw.
        (void)output_gradient; // Mark as unused to prevent compiler warnings
        throw std::runtime_error("ConvolutionLayer::backward() not implemented yet.");
        
        // If we were to return a dummy tensor of expected input grad shape:
        // This requires knowing the original input dimensions. If we assume output_gradient
        // has same N as original input, and we know original C, H, W from construction/last forward:
        // Tensor input_gradient(output_gradient.getN(), input_channels_, H_of_original_input, W_of_original_input);
        // return input_gradient; // This would be zero-initialized.
    }

    // Accessors for weights and biases (e.g., for external updates or inspection)
    Tensor& getWeights() { return weights_; }
    const Tensor& getWeights() const { return weights_; }
    Tensor& getBiases() { return biases_; }
    const Tensor& getBiases() const { return biases_; }

private:
    int input_channels_;
    int num_filters_;
    int kernel_size_;
    int stride_;
    int padding_;

    Tensor weights_; // Shape: (num_filters, input_channels, kernel_size, kernel_size)
    Tensor biases_;  // Shape: (1, num_filters, 1, 1) for easy broadcasting via at(0,f,0,0)
};


// --- MaxPoolingLayer ---
#include <limits> // Required for std::numeric_limits

class MaxPoolingLayer : public Layer {
public:
    MaxPoolingLayer(int pool_size, int stride = -1, std::string name = "MaxPoolingLayer")
        : Layer(std::move(name)),
          pool_size_(pool_size),
          stride_((stride == -1) ? pool_size : stride)
          // mask_ can be initialized here if needed, e.g., mask_(0,0,0,0) or based on expected input
    {
        if (pool_size_ <= 0) {
            throw std::invalid_argument("MaxPoolingLayer: Pool size must be positive.");
        }
        if (stride_ <= 0) {
            throw std::invalid_argument("MaxPoolingLayer: Stride must be positive.");
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input Validation
        if (input.getN() == 0 || input.getC() == 0 || input.getH() == 0 || input.getW() == 0) {
            throw std::invalid_argument("MaxPoolingLayer: Input tensor dimensions cannot be zero.");
        }
         if (input.getN() < 1 || input.getC() < 1 || input.getH() < 1 || input.getW() < 1) { // Redundant due to Tensor constructor, but good check
             throw std::invalid_argument("MaxPoolingLayer: Input tensor dimensions must be positive.");
        }


        // Output Dimensions Calculation
        int N = input.getN();
        int C = input.getC(); // Max pooling is typically done per-channel
        int input_H = input.getH();
        int input_W = input.getW();

        // Ensure input is large enough for at least one pool operation
        if (input_H < pool_size_ || input_W < pool_size_) {
            throw std::invalid_argument("MaxPoolingLayer: Input dimensions (H=" + std::to_string(input_H) +
                                        ", W=" + std::to_string(input_W) +
                                        ") are smaller than pool size (" + std::to_string(pool_size_) + ").");
        }

        int output_H = (input_H - pool_size_) / stride_ + 1;
        int output_W = (input_W - pool_size_) / stride_ + 1;

        if (output_H <= 0 || output_W <= 0) {
            throw std::invalid_argument("MaxPoolingLayer: Calculated output dimensions are non-positive. "
                                        "Output H: " + std::to_string(output_H) + 
                                        ", Output W: " + std::to_string(output_W));
        }

        // Output Tensor Creation
        Tensor output(N, C, output_H, output_W);
        
        // Optional: Initialize mask_ here if you were to implement it for backward pass
        // mask_ = Tensor(N, C, input_H, input_W); // Or shape of output, storing indices

        // Max Pooling Operation
        for (int n_idx = 0; n_idx < N; ++n_idx) { // Batch
            for (int c_idx = 0; c_idx < C; ++c_idx) { // Channels
                for (int oh_idx = 0; oh_idx < output_H; ++oh_idx) { // Output Height
                    for (int ow_idx = 0; ow_idx < output_W; ++ow_idx) { // Output Width
                        
                        float max_val = -std::numeric_limits<float>::infinity();
                        // int max_idx_kh = -1, max_idx_kw = -1; // For storing argmax for backward pass

                        int ih_start = oh_idx * stride_;
                        int iw_start = ow_idx * stride_;

                        for (int kh_idx = 0; kh_idx < pool_size_; ++kh_idx) { // Kernel Height
                            for (int kw_idx = 0; kw_idx < pool_size_; ++kw_idx) { // Kernel Width
                                
                                int current_ih = ih_start + kh_idx;
                                int current_iw = iw_start + kw_idx;

                                // Boundary check (should not be strictly necessary if output dims are calculated correctly
                                // and input_H/W >= pool_size_, but as a safeguard for non-unit strides with partial pools)
                                if (current_ih < input_H && current_iw < input_W) {
                                    float current_val = input.at(n_idx, c_idx, current_ih, current_iw);
                                    if (current_val > max_val) {
                                        max_val = current_val;
                                        // For backward pass:
                                        // max_idx_kh = kh_idx; 
                                        // max_idx_kw = kw_idx;
                                    }
                                }
                            }
                        }
                        output.at(n_idx, c_idx, oh_idx, ow_idx) = max_val;
                        
                        // For backward pass (mask_ storing 1s at max location in input window):
                        // if (mask_.getData()) { // Check if mask is initialized
                        //    int original_max_ih = ih_start + max_idx_kh;
                        //    int original_max_iw = iw_start + max_idx_kw;
                        //    if(max_idx_kh != -1) { // Ensure a max was found
                        //        mask_.at(n_idx, c_idx, original_max_ih, original_max_iw) = 1.0f;
                        //    }
                        // }
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        (void)output_gradient; // Mark as unused
        // A real implementation would use the stored mask_ (argmax indices) from the forward pass
        // to route the gradients back to the correct input locations.
        // Gradients for non-max locations are zero.
        throw std::runtime_error("MaxPoolingLayer::backward() not implemented yet.");
    }

private:
    int pool_size_;
    int stride_;
    // Tensor mask_; // For storing argmax indices for backward pass. Shape could be input shape or output shape.
};


// --- AveragePoolingLayer ---
class AveragePoolingLayer : public Layer {
public:
    AveragePoolingLayer(int pool_size, int stride = -1, std::string name = "AveragePoolingLayer")
        : Layer(std::move(name)),
          pool_size_(pool_size),
          stride_((stride == -1) ? pool_size : stride)
    {
        if (pool_size_ <= 0) {
            throw std::invalid_argument("AveragePoolingLayer: Pool size must be positive.");
        }
        if (stride_ <= 0) {
            throw std::invalid_argument("AveragePoolingLayer: Stride must be positive.");
        }
    }

    Tensor forward(const Tensor& input) override {
        // Input Validation
        if (input.getN() == 0 || input.getC() == 0 || input.getH() == 0 || input.getW() == 0) {
            throw std::invalid_argument("AveragePoolingLayer: Input tensor dimensions cannot be zero.");
        }
        if (input.getN() < 1 || input.getC() < 1 || input.getH() < 1 || input.getW() < 1) {
             throw std::invalid_argument("AveragePoolingLayer: Input tensor dimensions must be positive.");
        }

        // Output Dimensions Calculation
        int N = input.getN();
        int C = input.getC(); // Pooling is done per-channel
        int input_H = input.getH();
        int input_W = input.getW();

        // Ensure input is large enough for at least one pool operation
        if (input_H < pool_size_ || input_W < pool_size_) {
            throw std::invalid_argument("AveragePoolingLayer: Input dimensions (H=" + std::to_string(input_H) +
                                        ", W=" + std::to_string(input_W) +
                                        ") are smaller than pool size (" + std::to_string(pool_size_) + ").");
        }

        int output_H = (input_H - pool_size_) / stride_ + 1;
        int output_W = (input_W - pool_size_) / stride_ + 1;

        if (output_H <= 0 || output_W <= 0) {
            throw std::invalid_argument("AveragePoolingLayer: Calculated output dimensions are non-positive. "
                                        "Output H: " + std::to_string(output_H) + 
                                        ", Output W: " + std::to_string(output_W));
        }

        // Output Tensor Creation
        Tensor output(N, C, output_H, output_W);

        // Average Pooling Operation
        float pool_area = static_cast<float>(pool_size_ * pool_size_);

        for (int n_idx = 0; n_idx < N; ++n_idx) { // Batch
            for (int c_idx = 0; c_idx < C; ++c_idx) { // Channels
                for (int oh_idx = 0; oh_idx < output_H; ++oh_idx) { // Output Height
                    for (int ow_idx = 0; ow_idx < output_W; ++ow_idx) { // Output Width
                        
                        float sum = 0.0f;
                        int ih_start = oh_idx * stride_;
                        int iw_start = ow_idx * stride_;

                        for (int kh_idx = 0; kh_idx < pool_size_; ++kh_idx) { // Kernel Height
                            for (int kw_idx = 0; kw_idx < pool_size_; ++kw_idx) { // Kernel Width
                                
                                int current_ih = ih_start + kh_idx;
                                int current_iw = iw_start + kw_idx;
                                
                                // Boundary check (safeguard, similar to MaxPooling)
                                if (current_ih < input_H && current_iw < input_W) {
                                     sum += input.at(n_idx, c_idx, current_ih, current_iw);
                                }
                            }
                        }
                        output.at(n_idx, c_idx, oh_idx, ow_idx) = sum / pool_area;
                    }
                }
            }
        }
        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        (void)output_gradient; // Mark as unused
        // A real implementation would distribute the gradient evenly to all elements
        // in the pooling window of the input that contributed to an output element.
        throw std::runtime_error("AveragePoolingLayer::backward() not implemented yet.");
    }

private:
    int pool_size_;
    int stride_;
};


#endif // ML_LAYERS_H

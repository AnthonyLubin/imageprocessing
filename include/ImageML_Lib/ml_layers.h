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


// --- ReLULayer ---
#include <algorithm> // For std::max

class ReLULayer : public Layer {
public:
    ReLULayer(std::string name = "ReLULayer")
        : Layer(std::move(name)), input_cache_(0,0,0,0) /* Initialize with valid but empty tensor */ {}

    Tensor forward(const Tensor& input) override {
        // Store a deep copy of the input for the backward pass
        input_cache_ = input; // Relies on Tensor's copy assignment operator

        // Create output tensor with the same dimensions as input
        Tensor output(input.getN(), input.getC(), input.getH(), input.getW());

        const float* input_data = input.getData();
        float* output_data = output.getData();
        size_t total_elements = input.getSize();

        if (!input_data || !output_data) {
            throw std::runtime_error("ReLULayer: Input or output tensor data is null.");
        }

        for (size_t i = 0; i < total_elements; ++i) {
            output_data[i] = std::max(0.0f, input_data[i]);
        }

        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        // Validate shapes
        if (output_gradient.getN() != input_cache_.getN() ||
            output_gradient.getC() != input_cache_.getC() ||
            output_gradient.getH() != input_cache_.getH() ||
            output_gradient.getW() != input_cache_.getW()) {
            throw std::invalid_argument("ReLULayer::backward: output_gradient shape mismatch with input_cache_ shape.");
        }

        // Create input_gradient tensor with the same dimensions as input_cache_
        Tensor input_gradient(input_cache_.getN(), input_cache_.getC(), input_cache_.getH(), input_cache_.getW());

        const float* cached_data = input_cache_.getData();
        const float* grad_data = output_gradient.getData();
        float* input_grad_data = input_gradient.getData();
        size_t total_elements = input_cache_.getSize();

        if (!cached_data || !grad_data || !input_grad_data) {
            // This might happen if input_cache_ was not properly set (e.g. forward not called)
            // or if output_gradient is an invalid tensor.
            throw std::runtime_error("ReLULayer::backward: Tensor data is null.");
        }

        for (size_t i = 0; i < total_elements; ++i) {
            if (cached_data[i] > 0) {
                input_grad_data[i] = grad_data[i];
            } else {
                input_grad_data[i] = 0.0f;
            }
        }
        return input_gradient;
    }

private:
    Tensor input_cache_; // To store input tensor from forward pass for backward pass
};


// --- FlattenLayer ---
class FlattenLayer : public Layer {
public:
    FlattenLayer(std::string name = "FlattenLayer")
        : Layer(std::move(name)),
          original_N_(0), original_C_(0), original_H_(0), original_W_(0) {}

    Tensor forward(const Tensor& input) override {
        if (input.getN() <= 0) { // N can be >0 even if total size is 0 if C,H,W are 0.
             throw std::invalid_argument("FlattenLayer::forward: Input batch size (N) must be positive.");
        }
        // Store original input shape for backward pass
        original_N_ = input.getN();
        original_C_ = input.getC();
        original_H_ = input.getH();
        original_W_ = input.getW();

        // Handle case where C, H, or W might be zero, leading to zero features.
        // The Tensor reshape logic should handle N*0*0*0 if original size is 0.
        // If N > 0 but C*H*W = 0, this means num_features will be 0.
        int num_features = original_C_ * original_H_ * original_W_;

        // Create a copy of the input tensor to work with.
        // The data itself doesn't change, only its interpretation (shape).
        Tensor output = input; // Uses Tensor's copy constructor

        // Reshape the output tensor.
        // If num_features is 0, reshape will expect N*0*1*1 == N*C*H*W.
        // This holds if C*H*W was 0.
        output.reshape(original_N_, num_features, 1, 1);

        return output;
    }

    Tensor backward(const Tensor& output_gradient) override {
        // Validate that forward pass has been called and stored a valid shape
        if (original_N_ <= 0) { // Check N as it's the batch dimension and must be >0 from forward.
            throw std::logic_error("FlattenLayer::backward() called before forward() or with invalid stored shape (original_N_ <= 0).");
        }

        int expected_features = original_C_ * original_H_ * original_W_;

        // Validate output_gradient shape
        if (output_gradient.getN() != original_N_ ||
            output_gradient.getC() != expected_features ||
            output_gradient.getH() != 1 ||
            output_gradient.getW() != 1) {
            throw std::invalid_argument(
                "FlattenLayer::backward(): output_gradient shape mismatch. Expected (" +
                std::to_string(original_N_) + ", " + std::to_string(expected_features) + ", 1, 1), got (" +
                std::to_string(output_gradient.getN()) + ", " + std::to_string(output_gradient.getC()) + ", " +
                std::to_string(output_gradient.getH()) + ", " + std::to_string(output_gradient.getW()) + ")."
            );
        }

        // Create a copy of the output_gradient to work with.
        // The data itself doesn't change, only its interpretation (shape).
        Tensor input_gradient = output_gradient; // Uses Tensor's copy constructor

        // Reshape the gradient back to the original input shape
        input_gradient.reshape(original_N_, original_C_, original_H_, original_W_);

        return input_gradient;
    }

private:
    int original_N_;
    int original_C_;
    int original_H_;
    int original_W_;
};


// --- DenseLayer ---
#include <cmath> // For sqrt, used in weight initialization

class DenseLayer : public Layer {
public:
    DenseLayer(int input_features, int output_features, std::string name = "DenseLayer")
        : Layer(std::move(name)),
          input_features_(input_features),
          output_features_(output_features),
          weights_(input_features, output_features, 1, 1), // Shape: (K, M, 1, 1)
          biases_(1, output_features, 1, 1),              // Shape: (1, M, 1, 1)
          input_cache_(0,0,0,0) // Initialize with valid but empty tensor
    {
        if (input_features <= 0 || output_features <= 0) {
            throw std::invalid_argument("DenseLayer: input_features and output_features must be positive.");
        }

        // Initialize weights_ with scaled random values (e.g., Xavier/Glorot variant)
        float* w_data = weights_.getData();
        size_t w_size = weights_.getSize();
        float scale = std::sqrt(2.0f / (input_features_ /* + output_features_ */)); // Simpler variant
        // float scale = std::sqrt(1.0f / input_features_); // Alternative
        for (size_t i = 0; i < w_size; ++i) {
            // Random values typically from a normal distribution, or uniform [-limit, limit]
            w_data[i] = ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * 2.0f * scale;
        }
        // Biases are already initialized to zero by Tensor constructor.
    }

    // Input shape: (N, input_features_, 1, 1)
    // Output shape: (N, output_features_, 1, 1)
    Tensor forward(const Tensor& input) override {
        // Validate input shape
        if (input.getC() != input_features_ || input.getH() != 1 || input.getW() != 1) {
            throw std::invalid_argument(
                "DenseLayer::forward: Input shape mismatch. Expected (N, " +
                std::to_string(input_features_) + ", 1, 1), got (" +
                std::to_string(input.getN()) + ", " + std::to_string(input.getC()) + ", " +
                std::to_string(input.getH()) + ", " + std::to_string(input.getW()) + ")."
            );
        }
        if (input.getN() == 0) {
            throw std::invalid_argument("DenseLayer::forward: Input batch size (N) cannot be zero.");
        }

        input_cache_ = input; // Deep copy for backward pass

        // Perform matrix multiplication: output = input * weights_
        // input: (N, K, 1, 1), weights_: (K, M, 1, 1) -> output: (N, M, 1, 1)
        Tensor output = input.matmul(weights_);

        // Add biases: output = output + biases_
        // output: (N, M, 1, 1), biases_: (1, M, 1, 1) -> output: (N, M, 1, 1)
        output = output + biases_; // Uses overloaded operator+ with broadcasting

        return output;
    }

    // output_gradient shape: (N, output_features_, 1, 1)
    Tensor backward(const Tensor& output_gradient) override {
        // Validate output_gradient shape
        if (output_gradient.getC() != output_features_ ||
            output_gradient.getH() != 1 || output_gradient.getW() != 1 ||
            output_gradient.getN() != input_cache_.getN()) { // N must match cached N
            throw std::invalid_argument(
                "DenseLayer::backward: output_gradient shape mismatch. Expected (N=" +
                std::to_string(input_cache_.getN()) + ", C=" + std::to_string(output_features_) + ", H=1, W=1), got (" +
                std::to_string(output_gradient.getN()) + ", " + std::to_string(output_gradient.getC()) + ", " +
                std::to_string(output_gradient.getH()) + ", " + std::to_string(output_gradient.getW()) + ")."
            );
        }
         if (input_cache_.getN() == 0) { // Forward pass must have happened correctly
            throw std::logic_error("DenseLayer::backward: input_cache_ is empty or invalid (N=0). Forward pass might not have been called or failed.");
        }


        // Gradient for Weights (dWeights): input_cache_transposed * output_gradient
        // input_cache_ (N, K, 1, 1) -> transpose -> (K, N, 1, 1)
        // output_gradient (N, M, 1, 1)
        // dWeights = (K, N, 1, 1) matmul (N, M, 1, 1) -> (K, M, 1, 1) which is weights_ shape.
        Tensor input_cache_transposed = input_cache_.transpose();
        Tensor weights_gradient = input_cache_transposed.matmul(output_gradient);
        // In a full framework, weights_gradient would be stored or used by an optimizer.
        // For example: this->weights_ -= learning_rate * weights_gradient; (needs operator-= and scalar multiplication)

        // Gradient for Biases (dBiases): sum output_gradient along N (batch) axis
        // output_gradient (N, M, 1, 1) -> sum_along_axis_N -> (1, M, 1, 1) which is biases_ shape.
        Tensor biases_gradient = output_gradient.sum_along_axis_N();
        // In a full framework, biases_gradient would be stored or used by an optimizer.
        // For example: this->biases_ -= learning_rate * biases_gradient;

        // Gradient for Input (dInput / input_gradient): output_gradient * weights_transposed
        // weights_ (K, M, 1, 1) -> transpose -> (M, K, 1, 1)
        // output_gradient (N, M, 1, 1)
        // dInput = (N, M, 1, 1) matmul (M, K, 1, 1) -> (N, K, 1, 1) which is input_cache_ shape.
        Tensor weights_transposed = weights_.transpose();
        Tensor input_gradient = output_gradient.matmul(weights_transposed);

        return input_gradient;
    }

    // Accessors for parameters (optional, e.g., for inspection or external updates)
    const Tensor& getWeights() const { return weights_; }
    Tensor& getWeightsMutable() { return weights_; } // If direct modification is needed
    const Tensor& getBiases() const { return biases_; }
    Tensor& getBiasesMutable() { return biases_; }

private:
    int input_features_;
    int output_features_;
    Tensor weights_;      // Shape: (input_features, output_features, 1, 1)
    Tensor biases_;       // Shape: (1, output_features, 1, 1)
    Tensor input_cache_;  // Shape: (N, input_features, 1, 1)
};


#endif // ML_LAYERS_H

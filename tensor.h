#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept> // For std::out_of_range, std::runtime_error
#include <cstring>   // For memset, memcpy
#include <numeric>   // For std::accumulate (potentially useful for product of dims)
#include <algorithm> // For std::copy

class Tensor {
public:
    // Constructor: allocates memory for an NCHW tensor
    Tensor(int n, int c, int h, int w) : n_(n), c_(c), h_(h), w_(w) {
        if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive.");
        }
        size_t total_elements = static_cast<size_t>(n_) * c_ * h_ * w_;
        if (total_elements == 0) { // Should be caught by above, but as safeguard
             data_ = nullptr; // Or throw, depending on desired strictness for zero-sized tensors
        } else {
            data_ = new float[total_elements];
            if (!data_) {
                throw std::runtime_error("Failed to allocate memory for Tensor data.");
            }
            // Initialize data to zeros
            std::memset(data_, 0, total_elements * sizeof(float));
        }
    }

    // Destructor: frees allocated memory
    ~Tensor() {
        delete[] data_;
        data_ = nullptr;
    }

    // Copy Constructor (deep copy)
    Tensor(const Tensor& other) : n_(other.n_), c_(other.c_), h_(other.h_), w_(other.w_) {
        size_t total_elements = getSize();
        if (total_elements > 0) {
            data_ = new float[total_elements];
            if (!data_) {
                throw std::runtime_error("Failed to allocate memory for Tensor copy.");
            }
            std::copy(other.data_, other.data_ + total_elements, data_);
        } else {
            data_ = nullptr;
        }
    }

    // Copy Assignment Operator (deep copy)
    Tensor& operator=(const Tensor& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        // Free existing resource
        delete[] data_;

        // Copy dimensions
        n_ = other.n_;
        c_ = other.c_;
        h_ = other.h_;
        w_ = other.w_;

        // Allocate and copy new resource
        size_t total_elements = getSize();
        if (total_elements > 0) {
            data_ = new float[total_elements];
            if (!data_) {
                throw std::runtime_error("Failed to allocate memory for Tensor assignment.");
            }
            std::copy(other.data_, other.data_ + total_elements, data_);
        } else {
            data_ = nullptr;
        }
        return *this;
    }
    
    // Move Constructor
    Tensor(Tensor&& other) noexcept : data_(other.data_), n_(other.n_), c_(other.c_), h_(other.h_), w_(other.w_) {
        // Release the data pointer from the source object so it's not deleted by its destructor
        other.data_ = nullptr;
        other.n_ = 0;
        other.c_ = 0;
        other.h_ = 0;
        other.w_ = 0;
    }

    // Move Assignment Operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        // Free existing resource
        delete[] data_;

        // Pilfer the resources from the source object
        data_ = other.data_;
        n_ = other.n_;
        c_ = other.c_;
        h_ = other.h_;
        w_ = other.w_;

        // Release the data pointer from the source object so it's not deleted by its destructor
        other.data_ = nullptr;
        other.n_ = 0;
        other.c_ = 0;
        other.h_ = 0;
        other.w_ = 0;
        
        return *this;
    }


    // Accessors for dimensions
    int getN() const { return n_; }
    int getC() const { return c_; }
    int getH() const { return h_; }
    int getW() const { return w_; }

    // Get total number of elements
    size_t getSize() const {
        return static_cast<size_t>(n_) * c_ * h_ * w_;
    }

    // Access raw data pointer
    float* getData() { return data_; }
    const float* getData() const { return data_; }

    // Element access at (n, c, h, w)
    // Using row-major order for NCHW: index = n*C*H*W + c*H*W + h*W + w
    // More robustly: index = n * (C*H*W) + c * (H*W) + h * (W) + w
    float& at(int n_idx, int c_idx, int h_idx, int w_idx) {
        if (n_idx < 0 || n_idx >= n_ ||
            c_idx < 0 || c_idx >= c_ ||
            h_idx < 0 || h_idx >= h_ ||
            w_idx < 0 || w_idx >= w_) {
            throw std::out_of_range("Tensor index out of range.");
        }
        return data_[(static_cast<size_t>(n_idx) * c_ * h_ * w_) +
                     (static_cast<size_t>(c_idx) * h_ * w_) +
                     (static_cast<size_t>(h_idx) * w_) +
                      w_idx];
    }

    const float& at(int n_idx, int c_idx, int h_idx, int w_idx) const {
        if (n_idx < 0 || n_idx >= n_ ||
            c_idx < 0 || c_idx >= c_ ||
            h_idx < 0 || h_idx >= h_ ||
            w_idx < 0 || w_idx >= w_) {
            throw std::out_of_range("Tensor index out of range.");
        }
        return data_[(static_cast<size_t>(n_idx) * c_ * h_ * w_) +
                     (static_cast<size_t>(c_idx) * h_ * w_) +
                     (static_cast<size_t>(h_idx) * w_) +
                      w_idx];
    }

private:
    float* data_;
    int n_; // Batch size
    int c_; // Channels
    int h_; // Height
    int w_; // Width
};

#endif // TENSOR_H

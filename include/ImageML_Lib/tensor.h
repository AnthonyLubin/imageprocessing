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

public:
    // --- New methods ---

    // Reshape the tensor in-place.
    // The total number of elements must remain the same.
    void reshape(int new_n, int new_c, int new_h, int new_w) {
        if (static_cast<size_t>(new_n) * new_c * new_h * new_w != getSize()) {
            throw std::invalid_argument("Tensor::reshape: new shape total size (" +
                                        std::to_string(static_cast<size_t>(new_n) * new_c * new_h * new_w) +
                                        ") must match old shape total size (" + std::to_string(getSize()) + ").");
        }
        if (new_n <= 0 || new_c <= 0 || new_h <= 0 || new_w <= 0) {
            // Allow reshaping to (0,0,0,0) if original size is 0
            if (getSize() == 0 && new_n == 0 && new_c == 0 && new_h == 0 && new_w == 0) {
                 // This is a no-op for a truly empty tensor.
            } else {
                throw std::invalid_argument("Tensor::reshape: new dimensions must be positive (or all zero if original size is zero).");
            }
        }
        n_ = new_n;
        c_ = new_c;
        h_ = new_h;
        w_ = new_w;
    }

    // Element-wise addition with broadcasting support for bias-like tensors.
    // Returns a new Tensor with the result.
    Tensor operator+(const Tensor& other) const {
        // Case 1: Identical shapes
        if (n_ == other.n_ && c_ == other.c_ && h_ == other.h_ && w_ == other.w_) {
            Tensor result(n_, c_, h_, w_);
            float* result_data = result.getData();
            const float* this_data = data_;
            const float* other_data = other.getData();
            size_t total_elements = getSize();
            for (size_t i = 0; i < total_elements; ++i) {
                result_data[i] = this_data[i] + other_data[i];
            }
            return result;
        }

        // Case 2: Broadcasting a bias-like tensor (e.g., other has shape 1xCx1x1 or 1x1x1xC or 1xCxHxW etc.)
        // Simplified broadcasting: Check if 'other' dimensions are 1 or match 'this' dimensions.
        // And 'this' and 'other' must have the same rank (4D).
        if ( (other.n_ == 1 || other.n_ == n_) &&
             (other.c_ == 1 || other.c_ == c_) &&
             (other.h_ == 1 || other.h_ == h_) &&
             (other.w_ == 1 || other.w_ == w_) ) {

            Tensor result(n_, c_, h_, w_); // Output has the shape of the larger tensor ('this')

            for (int N_idx = 0; N_idx < n_; ++N_idx) {
                for (int C_idx = 0; C_idx < c_; ++C_idx) {
                    for (int H_idx = 0; H_idx < h_; ++H_idx) {
                        for (int W_idx = 0; W_idx < w_; ++W_idx) {
                            // Determine index for 'other' tensor, applying broadcasting (use 0 if dim is 1)
                            int other_N_idx = (other.n_ == 1) ? 0 : N_idx;
                            int other_C_idx = (other.c_ == 1) ? 0 : C_idx;
                            int other_H_idx = (other.h_ == 1) ? 0 : H_idx;
                            int other_W_idx = (other.w_ == 1) ? 0 : W_idx;

                            result.at(N_idx, C_idx, H_idx, W_idx) =
                                this->at(N_idx, C_idx, H_idx, W_idx) +
                                other.at(other_N_idx, other_C_idx, other_H_idx, other_W_idx);
                        }
                    }
                }
            }
            return result;
        }

        throw std::invalid_argument(
            "Tensor::operator+: shapes not compatible for addition/broadcasting. "
            "LHS shape: (" + std::to_string(n_) + "," + std::to_string(c_) + "," + std::to_string(h_) + "," + std::to_string(w_) + ") "
            "RHS shape: (" + std::to_string(other.n_) + "," + std::to_string(other.c_) + "," + std::to_string(other.h_) + "," + std::to_string(other.w_) + ")"
        );
    }

    // In-place version of operator+
    Tensor& operator+=(const Tensor& other) {
        // Case 1: Identical shapes
        if (n_ == other.n_ && c_ == other.c_ && h_ == other.h_ && w_ == other.w_) {
            float* this_data = data_;
            const float* other_data = other.getData();
            size_t total_elements = getSize();
            for (size_t i = 0; i < total_elements; ++i) {
                this_data[i] += other_data[i];
            }
            return *this;
        }

        // Case 2: Broadcasting (simplified for bias addition)
        if ( (other.n_ == 1 || other.n_ == n_) &&
             (other.c_ == 1 || other.c_ == c_) &&
             (other.h_ == 1 || other.h_ == h_) &&
             (other.w_ == 1 || other.w_ == w_) ) {

            for (int N_idx = 0; N_idx < n_; ++N_idx) {
                for (int C_idx = 0; C_idx < c_; ++C_idx) {
                    for (int H_idx = 0; H_idx < h_; ++H_idx) {
                        for (int W_idx = 0; W_idx < w_; ++W_idx) {
                            int other_N_idx = (other.n_ == 1) ? 0 : N_idx;
                            int other_C_idx = (other.c_ == 1) ? 0 : C_idx;
                            int other_H_idx = (other.h_ == 1) ? 0 : H_idx;
                            int other_W_idx = (other.w_ == 1) ? 0 : W_idx;

                            this->at(N_idx, C_idx, H_idx, W_idx) +=
                                other.at(other_N_idx, other_C_idx, other_H_idx, other_W_idx);
                        }
                    }
                }
            }
            return *this;
        }

        throw std::invalid_argument(
            "Tensor::operator+=: shapes not compatible for addition/broadcasting. "
            "LHS shape: (" + std::to_string(n_) + "," + std::to_string(c_) + "," + std::to_string(h_) + "," + std::to_string(w_) + ") "
            "RHS shape: (" + std::to_string(other.n_) + "," + std::to_string(other.c_) + "," + std::to_string(other.h_) + "," + std::to_string(other.w_) + ")"
        );
    }

    // Matrix multiplication: this (A) * other (B)
    // Assumes 'this' is A (N, K, 1, 1) and 'other' is B (K, M, 1, 1)
    // Result C will be (N, M, 1, 1)
    Tensor matmul(const Tensor& other) const {
        // Validate that H and W are 1 for both tensors (they are effectively 2D matrices)
        if (h_ != 1 || w_ != 1 || other.h_ != 1 || other.w_ != 1) {
            throw std::invalid_argument(
                "Tensor::matmul: Both tensors must have H=1 and W=1 to be treated as matrices. "
                "LHS shape: (" + std::to_string(n_) + "," + std::to_string(c_) + "," + std::to_string(h_) + "," + std::to_string(w_) + ") "
                "RHS shape: (" + std::to_string(other.n_) + "," + std::to_string(other.c_) + "," + std::to_string(other.h_) + "," + std::to_string(other.w_) + ")"
            );
        }

        // Matrix A dimensions: rowsA = n_, colsA = c_
        // Matrix B dimensions: rowsB = other.n_, colsB = other.c_
        int rowsA = n_;
        int colsA = c_;
        int rowsB = other.n_;
        int colsB = other.c_;

        // Validate matrix multiplication compatibility: colsA must equal rowsB
        if (colsA != rowsB) {
            throw std::invalid_argument(
                "Tensor::matmul: Incompatible matrix dimensions for multiplication. "
                "LHS (A) cols (" + std::to_string(colsA) + ") must match RHS (B) rows (" + std::to_string(rowsB) + "). "
                "LHS shape: (" + std::to_string(n_) + "," + std::to_string(c_) + ",1,1) "
                "RHS shape: (" + std::to_string(other.n_) + "," + std::to_string(other.c_) + ",1,1)"
            );
        }

        // Output tensor C dimensions: (rowsA, colsB, 1, 1)
        Tensor output(rowsA, colsB, 1, 1);

        // Perform matrix multiplication
        for (int i = 0; i < rowsA; ++i) {         // Iterate over rows of A (and output C)
            for (int j = 0; j < colsB; ++j) {     // Iterate over columns of B (and output C)
                float sum = 0.0f;
                for (int k = 0; k < colsA; ++k) { // Iterate over columns of A / rows of B (common dimension)
                    // For A: access element at (i, k) considering H=1, W=1
                    // For B: access element at (k, j) considering H=1, W=1
                    sum += this->at(i, k, 0, 0) * other.at(k, j, 0, 0);
                }
                output.at(i, j, 0, 0) = sum;
            }
        }
        return output;
    }

    // Transpose a tensor interpreted as a 2D matrix (H=1, W=1).
    // Swaps N (rows) and C (columns). Output shape: (C, N, 1, 1).
    Tensor transpose() const {
        if (h_ != 1 || w_ != 1) {
            throw std::logic_error(
                "Tensor::transpose: Tensor must have H=1 and W=1 to be treated as a 2D matrix for transpose. "
                "Current shape: (" + std::to_string(n_) + "," + std::to_string(c_) + "," +
                std::to_string(h_) + "," + std::to_string(w_) + ")"
            );
        }

        int original_rows = n_; // N
        int original_cols = c_; // C
        Tensor output(original_cols, original_rows, 1, 1); // New shape (C, N, 1, 1)

        for (int i = 0; i < original_rows; ++i) {    // Iterate original rows (N)
            for (int j = 0; j < original_cols; ++j) { // Iterate original columns (C)
                output.at(j, i, 0, 0) = this->at(i, j, 0, 0);
            }
        }
        return output;
    }

    // Sums tensor elements along the N (batch) dimension.
    // Output shape: (1, C, H, W).
    Tensor sum_along_axis_N() const {
        if (n_ == 0 && c_ == 0 && h_ == 0 && w_ == 0 && data_ == nullptr) { // Special case for uninitialized tensor
             return Tensor(0,0,0,0); // Return an empty tensor
        }
        if (n_ == 0) { // Cannot sum along N if N is 0 but other dims might not be
            throw std::logic_error("Tensor::sum_along_axis_N: Cannot sum along N-axis if N is 0.");
        }

        Tensor output(1, c_, h_, w_); // Output has N=1
        // Output is initialized to zeros by Tensor constructor.

        for (int n_idx = 0; n_idx < n_; ++n_idx) {
            for (int c_idx = 0; c_idx < c_; ++c_idx) {
                for (int h_idx = 0; h_idx < h_; ++h_idx) {
                    for (int w_idx = 0; w_idx < w_; ++w_idx) {
                        output.at(0, c_idx, h_idx, w_idx) += this->at(n_idx, c_idx, h_idx, w_idx);
                    }
                }
            }
        }
        return output;
    }
};

#endif // TENSOR_H

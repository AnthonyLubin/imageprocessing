# C++ Image Processing and Machine Learning Layers Library

## Overview
This project provides two main sets of functionalities:
1.  A basic C++ library for common image processing operations.
2.  A set of foundational layers for building simple Machine Learning (Deep Learning) models.

The implementations are kept straightforward for educational purposes.

---

## 1. Basic Image Processing Library

### Description
This part of the project offers tools to load, manipulate, and save images.

### Features
*   Load images (supports common formats like PNG, JPG, BMP, TGA via `stb_image`).
*   Save images (PNG, JPG, BMP, TGA via `stb_image_write`).
*   Convert images to **Grayscale**.
*   Apply a **Box Blur** filter.
*   Detect edges using the **Sobel Operator**.
*   Helper functions for creating blank images and freeing image memory.

### File Structure (Image Processing)
*   `image_processing.h`: Header file with declarations for the `Image` struct and image processing functions.
*   `image_processing.cpp`: Source file with implementations for `image_processing.h`.
*   `main.cpp`: Example program demonstrating image processing functions.
*   `stb_image.h` / `stb_image_write.h`: Public domain libraries for image I/O, included in `image_processing.cpp`.

### Dependencies (Image Processing)
*   `stb_image.h` and `stb_image_write.h` (included in the source).

### How to Compile (Image Processing Demo)
```bash
g++ main.cpp image_processing.cpp -o image_processor -std=c++11 -lm
```
*   `-std=c++11`: Specifies the C++11 standard.
*   `-lm`: Links the math library (may be needed for `cmath` functions like `sqrt`).

### How to Run (Image Processing Demo)
1.  **Prepare an input image:** Place `input.png` (or similar) in the build directory.
2.  **Run:** `./image_processor`
3.  **Output:** Files like `output_original.png`, `output_grayscale.png`, etc., will be generated.

### Basic Usage Example (Image Processing Library)
```cpp
#include "image_processing.h"
#include <iostream>

int main() {
    Image* my_image = load_image("my_photo.jpg");
    if (!my_image) { /* ... error handling ... */ return 1; }

    Image* gray_image = convert_to_grayscale(my_image);
    if (gray_image) {
        save_image(gray_image, "my_photo_grayscale.png");
        free_image(gray_image);
    }
    free_image(my_image);
    return 0;
}
```
Compile with: `g++ your_main.cpp image_processing.cpp -o your_app -std=c++11 -lm`

---

## 2. Machine Learning Layers

### Introduction
This part of the library provides foundational building blocks (layers) for creating simple neural networks. It includes a `Tensor` class for data representation and an abstract `Layer` interface.

### `tensor.h` - The `Tensor` Class
*   **Purpose:** A multi-dimensional array designed for machine learning data, typically used as a 4D structure: (Batch Size, Channels, Height, Width).
*   **Data Type:** Stores `float` values.
*   **Key Features:**
    *   Constructors to define shape and allocate memory (zero-initialized).
    *   Copy and Move constructors/assignment operators for proper memory management.
    *   `getData()`: Access to the raw data pointer.
    *   `getN()`, `getC()`, `getH()`, `getW()`: Methods to retrieve dimensions.
    *   `at(n, c, h, w)`: Element access with boundary checks.
    *   `getSize()`: Total number of elements.
    *   Destructor to free allocated memory.

### `ml_layers.h` - The `Layer` Interface and Implementations

#### Abstract Base Class: `Layer`
*   **Purpose:** Serves as an abstract base class for all neural network layers, defining a common interface.
*   **Key Virtual Methods:**
    *   `virtual Tensor forward(const Tensor& input) = 0;`: Defines the forward pass computation of the layer.
    *   `virtual Tensor backward(const Tensor& output_gradient) = 0;`: Defines the backward pass (backpropagation) computation. Currently, a placeholder in implemented layers.
*   **Other Features:**
    *   Stores a `layer_name_` (std::string) for identification.
    *   Virtual destructor for proper cleanup of derived classes.

#### Implemented Layers:

1.  **`ConvolutionLayer`**
    *   **Purpose:** Applies a 2D convolution operation, a core component of Convolutional Neural Networks (CNNs).
    *   **Inherits from:** `Layer`.
    *   **Constructor:** `ConvolutionLayer(int input_channels, int num_filters, int kernel_size, int stride = 1, int padding = 0, std::string name = "ConvolutionLayer")`
    *   **Parameters:**
        *   `input_channels`: Number of channels in the input tensor.
        *   `num_filters`: Number of convolution filters (determines output channels).
        *   `kernel_size`: Size of the square convolution kernel (e.g., 3 for 3x3).
        *   `stride`: Step size of the kernel movement (default: 1).
        *   `padding`: Zero-padding added to input borders (default: 0).
    *   **Weights & Biases:** Initializes its own weights (randomly) and biases (zeros). Weights are 4D (`num_filters`, `input_channels`, `kernel_size`, `kernel_size`), biases are 1D (broadcasted from `1, num_filters, 1, 1`).

2.  **`MaxPoolingLayer`**
    *   **Purpose:** Applies 2D max pooling, reducing spatial dimensions and retaining the most active features.
    *   **Inherits from:** `Layer`.
    *   **Constructor:** `MaxPoolingLayer(int pool_size, int stride = -1, std::string name = "MaxPoolingLayer")`
    *   **Parameters:**
        *   `pool_size`: Size of the square pooling window.
        *   `stride`: Step size of the window movement (default: `pool_size`).

3.  **`AveragePoolingLayer`**
    *   **Purpose:** Applies 2D average pooling, reducing spatial dimensions by averaging values in windows.
    *   **Inherits from:** `Layer`.
    *   **Constructor:** `AveragePoolingLayer(int pool_size, int stride = -1, std::string name = "AveragePoolingLayer")`
    *   **Parameters:**
        *   `pool_size`: Size of the square pooling window.
        *   `stride`: Step size of the window movement (default: `pool_size`).

### File Structure (Machine Learning)
*   `tensor.h`: Contains the `Tensor` class definition.
*   `ml_layers.h`: Contains the `Layer` abstract base class and concrete layer implementations (`ConvolutionLayer`, `MaxPoolingLayer`, `AveragePoolingLayer`).
*   `main_ml.cpp`: Example program demonstrating the usage of the ML layers.

### How to Compile (Machine Learning Demo)
The `main_ml.cpp` file demonstrates the ML layers.
```bash
# For the Machine Learning Layers demonstration
g++ main_ml.cpp image_processing.cpp -o main_ml -std=c++11 -I. -lm
```
*   `-I.` adds the current directory to the include path (if `tensor.h`, `ml_layers.h` are there).
*   `image_processing.cpp` is included in the command above. While `main_ml.cpp` itself doesn't directly use image processing functions, the `Tensor` or `Layer` classes might (even if unintentionally or transitively) depend on utilities or headers that were part of `image_processing.cpp` or its environment during development. If `tensor.h` and `ml_layers.h` are fully self-contained and have no dependencies on content from `image_processing.cpp` (like `stb_image` or specific image structures), you might simplify the compilation:
    ```bash
    # Alternative if image_processing.cpp is not a dependency for ML components
    # g++ main_ml.cpp -o main_ml -std=c++11 -I. -lm
    ```

### How to Run (Machine Learning Demo)
1.  **Compile `main_ml.cpp`** using the command above.
2.  **Run the executable:**
    ```bash
    ./main_ml
    ```
3.  **Expected Output:** The program will print to the console:
    *   The shape and sample values of an initial input tensor.
    *   The shape and sample values of the output tensor after passing through `ConvolutionLayer`.
    *   The shape and sample values of the output tensor after passing through `MaxPoolingLayer`.
    *   The shape and sample values of the output tensor after passing through `AveragePoolingLayer`.
    This demonstrates the data flow and dimension changes through the layers.

---
This README provides a guide to understanding, compiling, and using both the image processing and machine learning layer components of this library.

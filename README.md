# C++ Image Processing and Machine Learning Layers Library (ImageML_Lib)

## Overview
This project, `ImageML_Lib`, provides two main sets of functionalities:
1.  A basic C++ library for common image processing operations.
2.  A set of foundational layers for building simple Machine Learning (Deep Learning) models.

The implementations are kept straightforward for educational purposes. The project uses CMake for building and managing dependencies.

## File Structure
The project is organized as follows:
*   `CMakeLists.txt`: The main CMake build script.
*   `README.md`: This file.
*   `include/ImageML_Lib/`: Public header files for the library.
    *   `image_processing.h`: Declarations for image processing functions and `Image` struct.
    *   `tensor.h`: Definition of the `Tensor` class for ML.
    *   `ml_layers.h`: Abstract `Layer` class and concrete layer implementations (Convolution, Max Pooling, Average Pooling).
*   `src/`: Source code for the library implementation.
    *   `image_processing.cpp`: Implementation of image processing functions, including STB image library implementations.
*   `examples/`: Example programs demonstrating library usage.
    *   `main.cpp`: Demonstrates image processing functionalities.
    *   `main_ml.cpp`: Demonstrates machine learning layer functionalities.
*   `third_party/stb/`: Contains STB public domain header files.
    *   `stb_image.h`: For loading images.
    *   `stb_image_write.h`: For saving images.

---

## Building and Using the Library (CMake)

This project uses CMake for building the library and example applications.

### 1. Prerequisites
*   A C++ compiler supporting C++11 (e.g., GCC, Clang, MSVC).
*   CMake (version 3.10 or higher).
*   Make (or another build tool like Ninja, Visual Studio).

### 2. Building the Library and Examples
Follow these steps for an out-of-source build:

1.  **Navigate to the project root directory.** (Where this `README.md` and `CMakeLists.txt` are located).
2.  **Create a build directory and change into it:**
    ```bash
    mkdir build
    cd build
    ```
3.  **Run CMake to configure the project:**
    This command generates the build files for your chosen build system (e.g., Makefiles).
    ```bash
    cmake ..
    ```
    *   **Optional: Specify Install Location:** To install the library to a custom location (e.g., within your user directory), you can specify `CMAKE_INSTALL_PREFIX`:
        ```bash
        cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/your/custom/location
        ```
        If not specified, it defaults to system locations like `/usr/local` on Unix-like systems.

4.  **Compile the project:**
    Use the build tool specified by your CMake generator. For Makefiles (common on Linux/macOS):
    ```bash
    make
    ```
    Alternatively, you can use CMake's build tool mode, which is generator-agnostic:
    ```bash
    cmake --build .
    ```
    This will compile:
    *   The `ImageML` shared library (e.g., `libImageML.so` or `ImageML.dll`).
    *   The example executables: `ImageProcessorDemo` and `MLLayersDemo`.
    These build artifacts will be located within the `build` directory (e.g., library in `build/src/` or `build/`, examples in `build/examples/`).

### 3. Running the Examples

After successful compilation, you can run the example programs directly from the `build` directory:

*   **Image Processing Demo:**
    *   You'll need an image file (e.g., `input.png`) accessible by the executable. For simplicity, you can copy one into your `build` directory or provide a full path in `examples/main.cpp`.
    *   Run from `build` directory:
        ```bash
        ./examples/ImageProcessorDemo 
        ```
        This will generate output images (e.g., `output_original.png`) in the directory from which it's run (i.e., `build/`).

*   **Machine Learning Layers Demo:**
    *   Run from `build` directory:
        ```bash
        ./examples/MLLayersDemo
        ```
        This will print tensor shapes and sample values to the console, demonstrating the ML layer operations.

### 4. Installing the Library (Optional)

If you wish to install the library and headers to your system or the custom prefix specified during CMake configuration:

1.  **From within the `build` directory, run:**
    ```bash
    make install
    ```
    *   **Note:** `sudo` might be required if installing to default system locations (e.g., `sudo make install`). This is not needed if `CMAKE_INSTALL_PREFIX` was set to a user-writable path.

    This will typically install:
    *   The `ImageML` shared library to `[prefix]/lib/` (e.g., `/usr/local/lib/`).
    *   The public headers to `[prefix]/include/ImageML_Lib/` (e.g., `/usr/local/include/ImageML_Lib/`).
    *   The example executables to `[prefix]/bin/ImageML_Lib_examples/` (e.g., `/usr/local/bin/ImageML_Lib_examples/`).

### 5. Linking Against `ImageML_Lib` in Your CMake Project

Once `ImageML_Lib` is installed, you can use it in your own CMake project.

1.  **Ensure your project can find the installed library and headers.**
    *   If installed to a custom prefix, you may need to add this prefix to `CMAKE_PREFIX_PATH` when configuring your project:
        ```bash
        cmake -D CMAKE_PREFIX_PATH=/path/to/your/custom/location /path/to/your/project
        ```
    *   Alternatively, your project's `CMakeLists.txt` might need `find_package(ImageML_Lib)` if `ImageML_Lib` provided a CMake package configuration file (not implemented in this version). For now, direct linking is assumed.

2.  **Example `CMakeLists.txt` for your application:**
    ```cmake
    cmake_minimum_required(VERSION 3.10)
    project(MyImageApplication CXX)

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    # Option 1: If ImageML_Lib is installed in a standard system path 
    # or CMAKE_PREFIX_PATH is set correctly.
    # CMake should find headers and library automatically for linking.

    # Option 2: Manually specify paths if needed (less ideal than CMAKE_PREFIX_PATH)
    # include_directories(/path/to/your/custom/location/include) 
    # link_directories(/path/to/your/custom/location/lib)

    add_executable(MyImageApplication main.cpp)

    # Link against the ImageML library
    # The name "ImageML" is used as defined by add_library(ImageML ...)
    target_link_libraries(MyImageApplication PRIVATE ImageML)
    ```

3.  **In your C++ code, include the headers:**
    ```cpp
    #include <ImageML_Lib/image_processing.h> // For image functions
    #include <ImageML_Lib/tensor.h>           // For Tensor class
    #include <ImageML_Lib/ml_layers.h>      // For Layer classes
    #include <iostream>

    int main() {
        // Example: Using the Tensor class
        Tensor my_tensor(1, 3, 224, 224);
        std::cout << "Created a tensor of shape: " 
                  << my_tensor.getN() << "x" 
                  << my_tensor.getC() << "x" 
                  << my_tensor.getH() << "x" 
                  << my_tensor.getW() << std::endl;

        // Example: Load an image (if ImageML is linked)
        // Image* img = load_image("some_image.png");
        // if (img) {
        //     std::cout << "Loaded image with width: " << img->width << std::endl;
        //     free_image(img);
        // }
        return 0;
    }
    ```

---

## Library Components Details

### Image Processing (`image_processing.h`, `image_processing.cpp`)
*   **`Image` Struct:** Represents an image with `width`, `height`, `channels`, and `unsigned char* data`.
*   **Functions:**
    *   `load_image(const char* filename)`: Loads an image.
    *   `save_image(const Image* image, const char* filename)`: Saves an image.
    *   `free_image(Image* image)`: Frees image memory.
    *   `convert_to_grayscale(const Image* input_image)`
    *   `apply_blur_filter(const Image* input_image)`
    *   `detect_edges(const Image* input_image)`
*   **Dependencies:** Uses `stb_image.h` and `stb_image_write.h` (included) for image I/O.

### Machine Learning Layers (`tensor.h`, `ml_layers.h`)

#### `Tensor` Class (`tensor.h`)
*   **Purpose:** 4D tensor (Batch, Channels, Height, Width) storing `float` data.
*   **Features:** Constructors, destructor, data accessors (`getData`, `at`), dimension getters (`getN`, `getC`, `getH`, `getW`), copy/move semantics.

#### `Layer` Abstract Class (`ml_layers.h`)
*   **Interface:** Defines `virtual Tensor forward(...)` and `virtual Tensor backward(...)`.
*   **Naming:** Layers can be named for identification.

#### Concrete Layers (`ml_layers.h`):
1.  **`ConvolutionLayer`**:
    *   Applies 2D convolution.
    *   Constructor: `ConvolutionLayer(input_channels, num_filters, kernel_size, stride, padding, name)`
2.  **`MaxPoolingLayer`**:
    *   Applies 2D max pooling.
    *   Constructor: `MaxPoolingLayer(pool_size, stride, name)`
3.  **`AveragePoolingLayer`**:
    *   Applies 2D average pooling.
    *   Constructor: `AveragePoolingLayer(pool_size, stride, name)`

---
This README provides a guide to understanding, building, installing, and using the `ImageML_Lib` library.

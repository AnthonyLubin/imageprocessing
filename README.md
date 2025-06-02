# Simple C++ Image Processing Library

## Description
This project is a basic C++ library for performing common image processing operations. It provides functionalities to load, process, and save images. The implementations are kept simple for educational purposes.

## Features
*   Load images (supports common formats like PNG, JPG, BMP, TGA via `stb_image`).
*   Save images (PNG, JPG, BMP, TGA via `stb_image_write`).
*   Convert images to **Grayscale**.
*   Apply a **Box Blur** filter.
*   Detect edges using the **Sobel Operator**.
*   Helper functions for creating blank images and freeing image memory.

## File Structure
*   `image_processing.h`: The header file containing the declarations for the `Image` structure and all image processing functions. Include this file in your project to use the library.
*   `image_processing.cpp`: The source file containing the implementations of the functions declared in `image_processing.h`.
*   `main.cpp`: An example program that demonstrates how to use the various functions in the image processing library. It loads an image, applies several filters, and saves the results.
*   `stb_image.h`: Single-file public domain library for loading images. (Included in `image_processing.cpp`)
*   `stb_image_write.h`: Single-file public domain library for writing images. (Included in `image_processing.cpp`)

## Dependencies
This project relies on two public domain libraries:
*   `stb_image.h` for loading images.
*   `stb_image_write.h` for saving images.

These header files are directly included and implemented within `image_processing.cpp`. No external installation is required, but they must be present alongside `image_processing.cpp` or accessible in your include path if you modify the include directives.

## How to Compile
You can compile the project using a C++ compiler like g++. Ensure that `main.cpp` and `image_processing.cpp` are compiled together.

Example using g++:
```bash
g++ main.cpp image_processing.cpp -o image_processor -std=c++11 -lm
```
*   `-std=c++11`: Specifies the C++11 standard. You can use a newer standard if preferred.
*   `-lm`: Links the math library, which might be necessary for `sqrt` and other functions from `<cmath>`.

## How to Run
1.  **Prepare an input image:** Place an image file named `input.png` (or any other format supported by `stb_image`, like JPG, BMP) in the same directory where you compiled the `image_processor` executable. If you use a different filename, you'll need to modify it in `main.cpp`.
2.  **Run the executable:**
    ```bash
    ./image_processor
    ```
3.  **Check the output:** The program will generate the following output files in the same directory:
    *   `output_original.png`
    *   `output_grayscale.png`
    *   `output_blurred.png`
    *   `output_edges.png`

## Basic Usage Example (in your own code)

To use this library in your own C++ project:

1.  Make sure `image_processing.h` is in your include path and `image_processing.cpp` is part of your build.
2.  Include the header file:

```cpp
#include "image_processing.h"
#include <iostream>

int main() {
    // Load an image
    Image* my_image = load_image("my_photo.jpg");
    if (!my_image) {
        std::cerr << "Failed to load my_photo.jpg" << std::endl;
        return 1;
    }

    // Convert to grayscale
    Image* gray_image = convert_to_grayscale(my_image);
    if (gray_image) {
        save_image(gray_image, "my_photo_grayscale.png");
        std::cout << "Saved grayscale version as my_photo_grayscale.png" << std::endl;
        free_image(gray_image);
    } else {
        std::cerr << "Failed to convert to grayscale." << std::endl;
    }

    // Clean up
    free_image(my_image);

    return 0;
}

```
Then compile your project, linking `image_processing.cpp` (or its compiled object file).
For example: `g++ your_main.cpp image_processing.cpp -o your_app -std=c++11 -lm`

---
This README provides a basic guide to understanding, compiling, and using the image processing library.

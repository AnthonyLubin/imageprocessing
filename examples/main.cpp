#include <iostream>
#include "ImageML_Lib/image_processing.h"

int main() {
    // 1. Load an image
    const char *inputFilename = "input.png"; // Assuming an image file named input.png exists
    Image *original_image = load_image(inputFilename);

    if (!original_image) {
        std::cerr << "Error loading image '" << inputFilename << "'." << std::endl;
        std::cerr << "Please ensure that '" << inputFilename << "' exists in the same directory as the executable," << std::endl;
        std::cerr << "and that the stb_image library is correctly functioning." << std::endl;
        return 1;
    }
    std::cout << "Image '" << inputFilename << "' loaded successfully." << std::endl;
    display_image(original_image);
    save_image(original_image, "output_original.png");
    std::cout << "Saved original image to output_original.png" << std::endl;


    // 2. Convert the loaded image to grayscale
    std::cout << "\nConverting to grayscale..." << std::endl;
    Image *grayscale_image = convert_to_grayscale(original_image);
    if (!grayscale_image) {
        std::cerr << "Error converting image to grayscale." << std::endl;
        free_image(original_image);
        return 1;
    }
    display_image(grayscale_image);
    save_image(grayscale_image, "output_grayscale.png");
    std::cout << "Saved grayscale image to output_grayscale.png" << std::endl;

    // 3. Apply a blur filter to the grayscale image
    std::cout << "\nApplying blur filter to grayscale image..." << std::endl;
    Image *blurred_image = apply_blur_filter(grayscale_image);
    if (!blurred_image) {
        std::cerr << "Error applying blur filter." << std::endl;
        free_image(original_image);
        free_image(grayscale_image);
        return 1;
    }
    display_image(blurred_image);
    save_image(blurred_image, "output_blurred.png");
    std::cout << "Saved blurred image to output_blurred.png" << std::endl;

    // 4. Detect edges in the blurred image
    std::cout << "\nDetecting edges in blurred image..." << std::endl;
    Image *edges_image = detect_edges(blurred_image);
    if (!edges_image) {
        std::cerr << "Error detecting edges." << std::endl;
        free_image(original_image);
        free_image(grayscale_image);
        free_image(blurred_image);
        return 1;
    }
    display_image(edges_image);
    save_image(edges_image, "output_edges.png");
    std::cout << "Saved edges image to output_edges.png" << std::endl;

    // 5. Free all images
    std::cout << "\nFreeing image memory..." << std::endl;
    free_image(original_image);
    free_image(grayscale_image);
    free_image(blurred_image);
    free_image(edges_image);
    std::cout << "All image memory freed." << std::endl;

    std::cout << "\nImage processing demonstration complete." << std::endl;
    std::cout << "Output files: output_original.png, output_grayscale.png, output_blurred.png, output_edges.png" << std::endl;

    return 0;
}

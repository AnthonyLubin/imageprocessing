#include "ImageML_Lib/image_processing.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring> // For memcpy
#include <stdexcept> // For std::runtime_error

// Define these before including stb_image.h and stb_image_write.h
// This is a common practice to enable their implementations.
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h" // Assuming third_party/ is an include path, and stb/ is within it
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h" // Assuming third_party/ is an include path, and stb/ is within it

// Helper function to allocate memory for an Image struct and its data
Image *allocate_image(int width, int height, int channels) {
    Image *img = new Image;
    if (!img) {
        return nullptr; // Failed to allocate Image struct
    }
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->data = new unsigned char[width * height * channels];
    if (!img->data) {
        delete img; // Failed to allocate image data
        return nullptr;
    }
    return img;
}

// Function to load an image from a file
Image *load_image(const char *filename) {
    int width, height, channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    Image *img = allocate_image(width, height, channels);
    if (!img) {
        stbi_image_free(data);
        return nullptr;
    }
    memcpy(img->data, data, width * height * channels);
    stbi_image_free(data);
    return img;
}

// Function to save an image to a file
void save_image(const Image *image, const char *filename) {
    if (!image || !image->data) {
        std::cerr << "Error: Cannot save a null image." << std::endl;
        return;
    }
    // stb_image_write supports PNG, BMP, TGA, JPG
    // We'll try to infer from filename, default to PNG.
    const char *ext = strrchr(filename, '.');
    if (ext && (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0)) {
        stbi_write_jpg(filename, image->width, image->height, image->channels, image->data, 90); // Quality 90
    } else if (ext && strcmp(ext, ".bmp") == 0) {
        stbi_write_bmp(filename, image->width, image->height, image->channels, image->data);
    } else if (ext && strcmp(ext, ".tga") == 0) {
        stbi_write_tga(filename, image->width, image->height, image->channels, image->data);
    } else { // Default to PNG
        stbi_write_png(filename, image->width, image->height, image->channels, image->data, image->width * image->channels);
    }
}

// Function to display an image (textual representation)
void display_image(const Image *image) {
    if (!image || !image->data) {
        std::cout << "Cannot display null image." << std::endl;
        return;
    }
    std::cout << "Image dimensions: " << image->width << "x" << image->height << ", Channels: " << image->channels << std::endl;
    // Print a few pixel values as a sample
    std::cout << "Sample pixel data (first few pixels):" << std::endl;
    for (int i = 0; i < std::min(10, image->width * image->height * image->channels); ++i) {
        std::cout << static_cast<int>(image->data[i]) << " ";
    }
    std::cout << std::endl;
}

// Function to create a blank image (data initialized to 0)
Image *create_blank_image(int width, int height, int channels) {
    Image *img = allocate_image(width, height, channels);
    if (img && img->data) {
        memset(img->data, 0, width * height * channels);
    }
    return img;
}

// Function to free image memory
void free_image(Image *image) {
    if (image) {
        if (image->data) {
            delete[] image->data;
            image->data = nullptr;
        }
        delete image;
    }
}

// Generic image processing function
Image *process_image(const Image *input_image, void (*process_func)(unsigned char *data, int width, int height, int channels)) {
    if (!input_image || !input_image->data || !process_func) {
        return nullptr;
    }
    Image *output_image = allocate_image(input_image->width, input_image->height, input_image->channels);
    if (!output_image) {
        return nullptr;
    }
    memcpy(output_image->data, input_image->data, input_image->width * input_image->height * input_image->channels);
    process_func(output_image->data, output_image->width, output_image->height, output_image->channels);
    return output_image;
}

// Function to convert an image to grayscale
Image *convert_to_grayscale(const Image *input_image) {
    if (!input_image || !input_image->data) {
        return nullptr;
    }

    if (input_image->channels == 1) { // Already grayscale
        Image *output_image = allocate_image(input_image->width, input_image->height, 1);
        if (!output_image) return nullptr;
        memcpy(output_image->data, input_image->data, input_image->width * input_image->height);
        return output_image;
    }

    if (input_image->channels < 3) { // Not enough channels for standard RGB grayscale
        std::cerr << "Grayscale conversion requires at least 3 channels (RGB)." << std::endl;
        return nullptr;
    }

    Image *output_image = allocate_image(input_image->width, input_image->height, 1);
    if (!output_image) {
        return nullptr;
    }

    for (int y = 0; y < input_image->height; ++y) {
        for (int x = 0; x < input_image->width; ++x) {
            unsigned char *p_in = input_image->data + (y * input_image->width + x) * input_image->channels;
            unsigned char *p_out = output_image->data + (y * output_image->width + x);
            // Luminosity method: 0.299*R + 0.587*G + 0.114*B
            float gray = 0.299f * p_in[0] + 0.587f * p_in[1] + 0.114f * p_in[2];
            *p_out = static_cast<unsigned char>(gray);
        }
    }
    return output_image;
}

// Function to apply a blur filter (simple box blur 3x3)
Image *apply_blur_filter(const Image *input_image) {
    if (!input_image || !input_image->data) {
        return nullptr;
    }

    Image *output_image = allocate_image(input_image->width, input_image->height, input_image->channels);
    if (!output_image) {
        return nullptr;
    }

    int w = input_image->width;
    int h = input_image->height;
    int c = input_image->channels;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                float sum = 0;
                int count = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int nx = x + kx;
                        int ny = y + ky;
                        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                            sum += input_image->data[(ny * w + nx) * c + ch];
                            count++;
                        }
                    }
                }
                output_image->data[(y * w + x) * c + ch] = static_cast<unsigned char>(sum / count);
            }
        }
    }
    return output_image;
}

void fliphorizontal(Image *input_image){
    int w = input_image->width;
    int h = input_image->height;
    int Image_Data = input_image->data;
    int c = input_image->channels;
    for(int y = 0; y<h; ++i){
        for(int x = 0; x<w/2; ++j){
            for(int ch = 0; ch<c; ++ch){
                int left_idx = (y*w+x)*c + ch;
                int right_idx = (y*w+(w-1-x)) * c + ch; 
                std::swap(Image_Data[left_idx], Image_Data[right_idx);
        }
        }
    }

// Function to detect edges (Sobel operator)
Image *detect_edges(const Image *input_image) {
    if (!input_image || !input_image->data) {
        return nullptr;
    }

    // First, convert to grayscale if not already
    Image *gray_image;
    if (input_image->channels >= 3) {
        gray_image = convert_to_grayscale(input_image);
        if (!gray_image) return nullptr;
    } else if (input_image->channels == 1) {
        // Create a copy to work on, or work directly if source can be temp
        gray_image = allocate_image(input_image->width, input_image->height, 1);
        if(!gray_image) return nullptr;
        memcpy(gray_image->data, input_image->data, input_image->width * input_image->height);
    } else {
        std::cerr << "Edge detection requires a grayscale or RGB image." << std::endl;
        return nullptr;
    }


    Image *output_image = create_blank_image(gray_image->width, gray_image->height, 1);
    if (!output_image) {
        free_image(gray_image);
        return nullptr;
    }

    int w = gray_image->width;
    int h = gray_image->height;

    // Sobel kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float sumX = 0;
            float sumY = 0;

            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel_val = gray_image->data[((y + ky) * w + (x + kx))];
                    sumX += pixel_val * Gx[ky + 1][kx + 1];
                    sumY += pixel_val * Gy[ky + 1][kx + 1];
                }
            }
            float magnitude = std::sqrt(sumX * sumX + sumY * sumY);
            output_image->data[y * w + x] = static_cast<unsigned char>(std::min(std::max(magnitude, 0.0f), 255.0f));
        }
    }

    free_image(gray_image); // Free the intermediate grayscale image
    return output_image;
}

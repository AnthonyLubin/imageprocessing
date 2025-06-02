#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

// This header file contains declarations for image processing functions.

// Structure to represent an image
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char *data;
} Image;

// Function to load an image from a file
Image *load_image(const char *filename);

// Function to save an image to a file
void save_image(const Image *image, const char *filename);

// Function to display an image
void display_image(const Image *image);

// Function to create a blank image
Image *create_blank_image(int width, int height, int channels);

// Function to free image memory
void free_image(Image *image);

// Example of a generic image processing function
// It creates a copy of the input image and applies the process_func to the new image data.
Image *process_image(const Image *input_image, void (*process_func)(unsigned char *data, int width, int height, int channels));

// Function to convert an image to grayscale
Image *convert_to_grayscale(const Image *input_image);

// Function to apply a blur filter to an image
Image *apply_blur_filter(const Image *input_image);

// Function to detect edges in an image
Image *detect_edges(const Image *input_image);

#endif // IMAGE_PROCESSING_H

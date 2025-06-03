// STB IMAGE HEADER - Placeholder
// Typically, you would paste the full content of stb_image.h here.
// For the purpose of this task, this placeholder is sufficient.
// The actual implementation is included in src/image_processing.cpp via #define STB_IMAGE_IMPLEMENTATION.

#ifndef STB_IMAGE_INCLUDE_STB_IMAGE_H
#define STB_IMAGE_INCLUDE_STB_IMAGE_H

// Define STBI_NO_STDIO if you don't want functions that use FILE *
// #define STBI_NO_STDIO

// Define STBI_ONLY_JPEG, STBI_ONLY_PNG, STBI_ONLY_BMP, STBI_ONLY_TGA, STBI_ONLY_GIF, STBI_ONLY_PSD, STBI_ONLY_HDR,
// STBI_ONLY_PIC, STBI_ONLY_PNM (.ppm and .pgm) to reduce code size.
// #define STBI_ONLY_PNG

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations used in image_processing.cpp
// Ensure these match the actual stb_image.h signatures if possible.
// The STB_IMAGE_IMPLEMENTATION in image_processing.cpp will provide the actual definitions.

typedef unsigned char stbi_uc;
typedef unsigned short stbi_us;

stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
stbi_uc *stbi_load_from_file(FILE *f, int *x, int *y, int *channels_in_file, int desired_channels);
// Add stbi_load_from_memory, stbi_load_from_callbacks if used.

void stbi_image_free(void *retval_from_stbi_load);

char const *stbi_failure_reason(void);

#ifdef __cplusplus
}
#endif

#endif // STB_IMAGE_INCLUDE_STB_IMAGE_H

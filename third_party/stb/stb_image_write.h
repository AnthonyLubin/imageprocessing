// STB IMAGE WRITE HEADER - Placeholder
// Typically, you would paste the full content of stb_image_write.h here.
// For the purpose of this task, this placeholder is sufficient.
// The actual implementation is included in src/image_processing.cpp via #define STB_IMAGE_WRITE_IMPLEMENTATION.

#ifndef INCLUDE_STB_IMAGE_WRITE_H
#define INCLUDE_STB_IMAGE_WRITE_H

#ifdef __cplusplus
extern "C" {
#endif

// Function declarations used in image_processing.cpp
// Ensure these match the actual stb_image_write.h signatures.
// The STB_IMAGE_WRITE_IMPLEMENTATION in image_processing.cpp will provide the definitions.

int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
// Add stbi_write_hdr if used

// If you use stbi_write_to_func, declare its callback type:
// typedef void stbi_write_func(void *context, void *data, int size);
// int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data, int stride_in_bytes);
// etc. for other formats

#ifdef __cplusplus
}
#endif

#endif // INCLUDE_STB_IMAGE_WRITE_H

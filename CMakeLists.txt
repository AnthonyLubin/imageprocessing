cmake_minimum_required(VERSION 3.10)
project(ImageML_Lib VERSION 1.0 LANGUAGES CXX)

# Include GNUInstallDirs for standard installation paths
include(GNUInstallDirs)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- ImageML Library Target ---
# Add sources for the library.
# For now, only image_processing.cpp contains actual compilable source code.
# tensor.h and ml_layers.h are header-only in terms of library definition,
# their content will be compiled as part of targets that include them (like executables).
add_library(ImageML SHARED 
    src/image_processing.cpp
    # If tensor.cpp or ml_layers.cpp with non-template/non-inline code existed, add them here.
)

# Include directories for the ImageML library
target_include_directories(ImageML 
    PUBLIC # For headers that consumers of the library need
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
        # This allows #include "ImageML_Lib/header.h"
    PRIVATE # For headers needed only to compile the library itself
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        # This allows #include "stb/stb_image.h" from within image_processing.cpp
)

# Link math library (libm) on Unix-like systems, excluding Apple (macOS/iOS)
# <cmath> functions (like sqrt, floor, ceil) often require linking against 'm'.
if(UNIX AND NOT APPLE)
    target_link_libraries(ImageML PRIVATE m)
endif()

# --- Example Executables ---

# 1. Image Processing Demo
add_executable(ImageProcessorDemo examples/main.cpp)
target_link_libraries(ImageProcessorDemo PRIVATE ImageML)
# By linking ImageML, ImageProcessorDemo inherits the PUBLIC include directories 
# of ImageML, so it can find "ImageML_Lib/image_processing.h".

# 2. Machine Learning Layers Demo
add_executable(MLLayersDemo examples/main_ml.cpp)
target_link_libraries(MLLayersDemo PRIVATE ImageML)
# Similarly, MLLayersDemo inherits ImageML's PUBLIC include directories,
# allowing it to find "ImageML_Lib/tensor.h" and "ImageML_Lib/ml_layers.h".


# --- Installation Rules ---

# Install the ImageML library target
# ARCHIVE for static libs or DLL import libs on Windows
# LIBRARY for .so/.dylib shared libs (non-Windows)
# RUNTIME for .dll shared libs (Windows) and executables
install(TARGETS ImageML
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
)

# Install public headers
# This installs the contents of include/ImageML_Lib/ to ${CMAKE_INSTALL_INCLUDEDIR}/ImageML_Lib
install(DIRECTORY include/ImageML_Lib/ 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ImageML_Lib
    FILES_MATCHING PATTERN "*.h" # Ensure only headers are installed
)

# Install example binaries (optional, but good for users)
install(TARGETS ImageProcessorDemo MLLayersDemo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ImageML_Lib_examples 
    # Installs to a subdirectory like /usr/local/bin/ImageML_Lib_examples
)

# Final status message
message(STATUS "CMake project ImageML_Lib configured with library, examples, and installation rules.")

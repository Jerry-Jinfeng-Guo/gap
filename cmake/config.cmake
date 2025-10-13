# GAP Power Flow Calculator Configuration

# Project version
set(GAP_VERSION_MAJOR 1)
set(GAP_VERSION_MINOR 0)
set(GAP_VERSION_PATCH 0)
set(GAP_VERSION "${GAP_VERSION_MAJOR}.${GAP_VERSION_MINOR}.${GAP_VERSION_PATCH}")

# Configure header file
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
    "${CMAKE_BINARY_DIR}/include/gap/config.h"
    @ONLY
)

# Add configured include directory
include_directories("${CMAKE_BINARY_DIR}/include")

# Package configuration
set(CPACK_PACKAGE_NAME "GAP")
set(CPACK_PACKAGE_VERSION_MAJOR ${GAP_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${GAP_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${GAP_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU-Accelerated Power Flow Calculator")
set(CPACK_PACKAGE_VENDOR "Your Organization")
set(CPACK_PACKAGE_CONTACT "your.email@example.com")

# CPack configuration
include(CPack)
#!/bin/bash
# Format all C++/CUDA files in the project
echo "Formatting C++/CUDA files with clang-format..."
find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | \
    xargs clang-format -i --style=file
echo "Formatting complete!"
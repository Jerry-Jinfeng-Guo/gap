#!/bin/bash
# Format all C++/CUDA files in the project using clang-format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Formatting C++/CUDA files with clang-format..."

# Option 1: Use pre-commit (if available)
if command -v pre-commit >/dev/null 2>&1 && [ -f .pre-commit-config.yaml ]; then
    echo "Using pre-commit clang-format..."
    pre-commit run clang-format --all-files
else
    # Option 2: Use clang-format directly
    echo "Using clang-format directly..."
    find . -name "*.cpp" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | \
        grep -v build/ | \
        xargs clang-format -i --style=file
fi

echo "Formatting complete!"
echo ""
echo "To check what changed, run:"
echo "  git diff"
echo ""
echo "To commit the changes, run:"
echo "  git add -A && git commit -m 'Format code with clang-format'"
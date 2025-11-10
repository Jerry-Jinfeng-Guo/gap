#!/bin/bash
# Format Python code with Black and isort

set -e

# Change to the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🎨 Running Python code formatters..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "📦 Using virtual environment: .venv"
else
    echo "⚠️  No virtual environment found. Make sure black and isort are installed globally."
fi

# Run black formatter
echo "🖤 Running Black formatter..."
python -m black --line-length=88 . || {
    echo "❌ Black formatting failed"
    exit 1
}

# Run isort import formatter
echo "📚 Running isort import formatter..."
python -m isort . || {
    echo "❌ isort formatting failed"
    exit 1
}

echo "✅ Python code formatting complete!"
echo "💡 Tip: Run 'pre-commit run --all-files' to check all formatters"
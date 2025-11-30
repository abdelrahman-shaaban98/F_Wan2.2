#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Wan2.2 Animate setup..."

# 1. Clone the repository
REPO_URL="https://github.com/Wan-Video/Wan2.2.git"
DIR_NAME="Wan2.2"

if [ -d "$DIR_NAME" ]; then
    echo "Directory $DIR_NAME already exists. Skipping clone."
    cd "$DIR_NAME"
else
    echo "Cloning repository..."
    git clone "$REPO_URL"
    cd "$DIR_NAME"
fi

# 2. Install Dependencies
echo "Installing dependencies..."

# Install Hugging Face CLI (required for downloading weights)
pip install "huggingface_hub[cli]"

# Install the main requirements
if [ -f "requirements.txt" ]; then
    echo "Installing base requirements..."
    pip install -r requirements.txt
fi

# Install the specific requirements for Wan-Animate
if [ -f "requirements_animate.txt" ]; then
    echo "Installing Wan-Animate specific requirements..."
    pip install -r requirements_animate.txt
else
    echo "Warning: requirements_animate.txt not found!"
fi

# --- FIX & EXTRA TOOLS ---
echo "Applying fixes and installing media tools..."

# Fix the 'No module named transformers.modeling_layers' error
pip install --upgrade peft transformers

# Install audio/video processing tools
pip install moviepy librosa
# -------------------------

# 3. Download Wan-Animate Weights
MODEL_ID="Wan-AI/Wan2.2-Animate-14B"
LOCAL_DIR="./Wan2.2-Animate-14B"

echo "Downloading weights for $MODEL_ID..."
echo "This may take a while depending on your internet connection..."

huggingface-cli download "$MODEL_ID" --local-dir "$LOCAL_DIR"

echo "----------------------------------------------------------------"
echo "Setup complete!"
echo "Weights are located in: $(pwd)/$LOCAL_DIR"
echo "Dependencies (including peft fix and moviepy) are installed."
echo "----------------------------------------------------------------"

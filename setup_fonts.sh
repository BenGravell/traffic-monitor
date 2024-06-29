#!/bin/bash

# Store the current directory
CURRENT_DIR=$(pwd)

# Define the download and installation directories
DOWNLOAD_DIR="fonts/IBM_Plex_Mono"

# Create the download directory
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

# Define an array of font files to download
FILES=(
  "IBMPlexMono-Bold.ttf"
  "IBMPlexMono-BoldItalic.ttf"
  "IBMPlexMono-Italic.ttf"
  "IBMPlexMono-Regular.ttf"
  "OFL.txt"
)

# Base URL for downloading the files from Google Fonts
BASE_URL="https://github.com/google/fonts/raw/main/ofl/ibmplexmono/"

# Loop over the files and download each one
for FILE in "${FILES[@]}"; do
  wget -q --show-progress -O "${FILE}" "${BASE_URL}${FILE}" -o /dev/null
done

echo "Fonts downloaded."

# Change back to the original directory
cd "$CURRENT_DIR"

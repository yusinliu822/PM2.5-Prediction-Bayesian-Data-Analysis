#!/bin/bash

# Set the directory containing the folders to unzip
directory="data"

# Change to the specified directory
cd "$directory"

# Loop through all folders within the directory
for folder in */; do
    # Get the folder name by removing the trailing slash
    folder_name="${folder%/}"

    # Unzip the folder
    unzip "$folder_name.zip"

    # Optional: Remove the zip file after extraction
    rm "$folder_name.zip"

done

for folder in */; do
    cd "$folder"
    for file in *.zip; do
        unzip "$file"
        rm "$file"
    done

    for file in *; do
        mv "$file" ..
    done

    cd ..
    rm -r "$folder"
done
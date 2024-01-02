dataPath="./external/arctic/data/arctic_data/data/images"
# dataPath="./data/rgb_data/ARCTIC"

# Ensure the parent folder exists
if [ ! -d "$dataPath" ]; then
  echo "Parent folder does not exist."
  exit 1
  else 
    echo "Parent folder exists."
fi

# Find all JPG files and organize them
find "$dataPath" -type f -name "*.jpg" | while read -r jpg_file; do
  # Extract the directory containing the JPG file
  jpg_dir="$(dirname "$jpg_file")"
  
  rgbFolder="${jpg_dir}/rgb"

  # Ensure the "rgb" folder exists
  mkdir -p "$rgbFolder"
  
  # Move the JPG file to the "rgb" folder
  mv "$jpg_file" "$rgbFolder"
done

echo "JPG files have been moved to their respective directories."

#!/bin/bash
# stringList="*gt_vis* *keypoints2d* *mediapipe_keypoints2d* *mmpose_vid* *pymafx_out* *rgb_mediapipe*"
dataPath="./data/rgb_data/DexYCB"

# # delete the folders
# for searchString in $stringList; do
#   for file in $(find $dataPath -iname "*$searchString*" -type d); do 
#     echo $file
#     rm -rf "$file"
#   done
#   echo "Images containing the search string '$searchString' have been removed."
# done

# Prompt the user to enter the search string
searchString="aligned_depth_to_color_"
folderName="depth"

# Find all image files with the given string in the current directory and its subdirectories
imageFiles=$(find $dataPath -iname "$searchString*.jpg" -o -iname "$searchString*.jpeg" -o -iname "$searchString*.png" -type f)

# Process each depth image file
for imageFile in $imageFiles; do
  # Get the parent directory of the image file
  parentDir=$(dirname "$imageFile")
  
  # Create a directory with the search string as its name within the parent directory
  directoryName="${parentDir}/${folderName}"
  mkdir -p "$directoryName"
  
  # Move the image file into the newly created directory
  mv "$imageFile" "$directoryName"
done
echo "Images containing the search string '$searchString' have been moved to their respective directories."

searchString1="color_"
folderName1="rgb"

# Find all image files with the given string in the current directory and its subdirectories
imageFiles1=$(find $dataPath -iname "$searchString1*.jpg" -o -iname "$searchString1*.jpeg" -o -iname "$searchString1*.png" -type f)

# Process each depth image file
for imageFile in $imageFiles1; do
  # Get the parent directory of the image file
  parentDir=$(dirname "$imageFile")
  
  # Create a directory with the search string as its name within the parent directory
  directoryName="${parentDir}/${folderName1}"
  mkdir -p "$directoryName"
  
  # Move the image file into the newly created directory
  mv "$imageFile" "$directoryName"
done
echo "Images containing the search string '$searchString1' have been moved to their respective directories."


searchString2="contact_labels_"
folderName2="contact_labels"

# Find all image files with the given string in the current directory and its subdirectories
imageFiles2=$(find $dataPath -iname "$searchString2*.json" -type f)

# Process each depth image file
for imageFile in $imageFiles2; do
  # Get the parent directory of the image file
  parentDir=$(dirname "$imageFile")
  
  # Create a directory with the search string as its name within the parent directory
  directoryName="${parentDir}/${folderName2}"
  mkdir -p "$directoryName"
  
  # Move the image file into the newly created directory
  mv "$imageFile" "$directoryName"
done

echo "JSONs containing the search string '$searchString2' have been moved to their respective directories."


searchString3="labels_"
folderName3="labels"

# Find all image files with the given string in the current directory and its subdirectories
imageFiles3=$(find $dataPath -iname "$searchString3*.npz" -type f)

# Process each depth image file
for imageFile in $imageFiles3; do
  # Get the parent directory of the image file
  parentDir=$(dirname "$imageFile")
  
  # Create a directory with the search string as its name within the parent directory
  directoryName="${parentDir}/${folderName3}"
  mkdir -p "$directoryName"
  
  # Move the image file into the newly created directory
  mv "$imageFile" "$directoryName"
done

echo "npz files containing the search string '$searchString3' have been moved to their respective directories."
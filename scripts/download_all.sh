#!/bin/bash
set -e

echo "Please enter your username in HMP webpage:"
# read USERNAME 
export USERNAME=$USERNAME

echo "Please enter your password"
# stty -echo
# read PASSO
export PASSWORD=$PASSWORD

mkdir -p downloads
zip_files=("body_models.zip" "model.zip" "data.zip")

# Loop through each zip file
for zip_file in "${zip_files[@]}"; do
    # Check if the zip file exists
    if [ -f "downloads/$zip_file" ]; then
        # Delete the zip file
        rm "downloads/$zip_file"
        echo "Deleted $directory/$zip_file"
    else
        echo "Zip file $zip_file does not exist in $directory"
    fi
done

hmp_path="./outputs/generative/results"
bm_path="./data/body_models"
pymafx_path="./external/PyMAF-X/data"
mmpose_path="./data/mmpose_models"

# create paths for data
mkdir -p $hmp_path $bm_path $pymafx_path $mmpose_path

echo "Downloading Body Models, PyMAF-X, and HMP"
python src/scripts/download_all.py 

echo "Downloading MMPose"
hand_detector_path="https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"
# keypoint_detector_path="https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-739c8639_20210330.pth"
keypoint_detector_path="https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth"

curl -L -o "$mmpose_path/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth" "$hand_detector_path"
curl -L -o "$mmpose_path/res50_onehand10k_256x256-e67998f6_20200813.pth" "$keypoint_detector_path"

unzip -o "downloads/body_models.zip" -d $bm_path
unzip -o "downloads/data.zip" -d $pymafx_path
unzip -o "downloads/model.zip" -d $hmp_path

echo "DONE!"




 
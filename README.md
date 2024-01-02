## Welcome to HMP! 

![Project Overview](overview.png)

**Disclaimer:** This code was developed on Ubuntu 20.04 with Python 3.10, CUDA 12.1 and PyTorch 2.1. 

## Setup 

### Conda Environment 

There is a single conda environment required to run the project (**hmp**). Requirements can be found in requirements.txt. To create the environment and install necessary packages:

```
source scripts/create_venv.sh
```

It installs all the necessary packages (including PyMAF-X related packages) to run the project. In case of installation problems, you may refer to related pages of [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md), [mmpose](https://mmpose.readthedocs.io/en/latest/installation.html). 


<!-- **(Optional)**, if you want to use mmpose as keypoint detector, you can install with:   

```
source scripts/install_mmpose.sh
``` -->


**Note:** There is no need to clone code from the git repos of [METRO](https://github.com/microsoft/MeshTransformer) or [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X). It is already incorporated under `./external/MeshTransformer` and `./external/PyMAF-X` 


### Download Models 
 We are done with setting the virtual environment up. Code is arranged. Now it is time to put body models, HMP model and initializator (PyMAF-X) model. Go to [HMP](https://hmp.is.tue.mpg.de) webpage to register. Then to download all required models, run:

```
source scripts/download_all.sh 
```

In the execution you should enter your email and password for the website. This will download all required models. In case of problems running script, you can manually donwload from our [website](https://hmp.is.tue.mpg.de).

**Note:** If you would like to use [METRO](https://github.com/microsoft/MeshTransformer) instead of [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X), Please follow the instructions on file downloads on their [website](https://github.com/microsoft/MeshTransformer/blob/main/docs/DOWNLOAD.md).  Folder orders are exactly same as suggested there. 

<!-- **Note:** In multi-stage optimization, we need to have a 2D keypoint detector. If you aim to use MMPose, download [hand detection](https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth) and [hand keypoint](https://download.openmmlab.com/mmpose/hand/resnet/res50_onehand10k_256x256-739c8639_20210330.pth) models. You can change models used in mmpose from [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo/hand_2d_keypoint.html#hand-2d-keypoint).  -->

## Training Motion Prior (Optional)

**Note: A pretrained version of the motion prior model is already downloaded  and placed under `./outputs/generative/results/model`. Follow this part only if you really want to train.** 

In motion prior training we used GRAB, TCDHands and SAMP from [AMASS](https://amass.is.tue.mpg.de/). Download the [ARCTIC](https://arctic.is.tue.mpg.de/) dataset also. Those raw datasets should be placed under `./data/amass_raw`

For processing the raw AMASS and ARCTIC datasets. Run the following commands respectively:  

```
python src/scripts/process_amass.py
python src/scripts/process_arctic.py
```

To split the processed dataset into train, test, and validation sets we run:

```
python src/datasets/amass.py amass.yaml
```

### Train on single motion sequence

To train the model on single-motion AMASS sequences, use:

```
python src/train_basic.py basic.yaml
```

The code will obtain sequences of 32, 64, 128, 256, and 512 frames and reconstruct them at 30, 60, 120, and 240 fps.

### Train generative 

```
python src/train.py generative.yaml
```

To evaluate the trained model, we can see its performance on motion_reconstruction, motion_inbetweening or latent_interpolation. You can run:

```
python src/application.py --config application.yaml --task TASKNAME --save_path SAVEPATH
```

For interactive visualization for those tasks, run ```python scenepic_viz.py SAVEPATH```.

## Hand Pose & Shape Estimation

### DexYCB & HO3D

If you want to run/evaluate HMP on them, download [DexYCB](https://dex-ycb.github.io/) and [HO3D-v3](https://www.tugraz.at/index.php?id=40231) datasets and put them in ```data/rgb_data/HO3D_v3``` and ```data/rgb_data/DexYCB``` respectively. 

For DexYCB, you first need to run ./scripts/process_dexycb.sh 

> We are intending to publish quantitative numbers for [ARCTIC](https://github.com/zc-alexfan/arctic). 

### Other datasets (in-the-wild videos)

You can download and try out new videos. Download a 30-fps video and put it in ```data/rgb_data/in_the_wild/FOLDERNAME/rgb_raw.mp4```. An example command for fitting:

```
python src/fitting_app.py --config CONFIGNAME.yaml --vid-path data/rgb_data/FOLDERNAME/rgb_raw.mp4
```

The output will be placed in `./optim/CONFIGNAME/DATASETNAME/FOLDERNAME/recon_000_30fps_hmp.mp4`. We share some sample configs (```in_the_lab_sample_config.yaml``` & ```in_the_wild_sample_config.yaml```) under `configs` folder for in_the_lab(HO3D, DexYCB, ARCTIC etc. ) and in_the_wild setting. You can play with the numbers.


To better illustrate the output structure please look at the folowing example. 

```
./data
    ├── rgb_data
    |   ├── DexYCB
    |   ├── HO3D_v3
    |   └── in_the_wild   
    ├── body_models 
    ├── mmpose_models
    ├── amass(*)
    ├── amass_raw(*)
    └── amass_processed_hist(*)
```
*Optional folders. `amass`, `amass_raw`, and `amass_processed_hist` folders are for training motion prior.

> For better results make sure that hands are not interacting and very close. This deteriorates bounding box detection and regression performance. 

## Folder overview

After collecting the above necessary files, the directory structure is expected as follows.  

```
├── data
|    ├── amass (*)
|    |   ├── generative 
|    |   └── single 
|    ├── amass_raw (*)
|    |   ├── GRAB 
|    |   ├── SAMP
|    |   └── TCDHands  
|    ├── body_models
|    |   ├── mano
|    |   |   ├── MANO_LEFT.pkl
|    |   |   └── MANO_RIGHT.pkl
|    |   └── smplx
|    |       └── SMPLX_NEUTRAL.npz
|    ├── rgb_data
|    |   ├── DexYCB
|    |   ├── HO3D_v3
|    |   └── in_the_wild 
|    └── mmpose models  
|
├── external
|    ├── mmpose
|    ├── PyMAF-X
|    ├── v2a_code
|    └── MeshTransformer
|
└── optim
    └── CONFIGNAME 
        └── DATASETNAME 
            └── FOLDERNAME


```
(*) Optional folders for motion prior training.

  
## Acknowledgements

- HMP codebase is adapted from [NEMF](https://github.com/c-he/NeMF) 
- Part of the code in `src/utils.py` and `python src/scripts/process_amass.py` is taken from [HuMoR](https://github.com/davrempe/humor/blob/b86c2d9faf7abd497749621821a5d46211304d62/humor/scripts/process_amass_data.py).
- We use [METRO](https://github.com/microsoft/MeshTransformer/tree/main) and [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X) as initialization methods and baselines
- We use [MediaPipe](https://developers.google.com/mediapipe) and [MMPose](https://github.com/open-mmlab/mmpose) as 2D keypoint sources. 
- We use [TempCLR](https://github.com/eth-ait/tempclr/tree/master/TempCLR) for quantitative evaluation. 
- We use [MANO](https://mano.is.tue.mpg.de/) hand model. 
- Shout-out to [manopth](https://github.com/hassony2/manopth/tree/4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b) and [transformers](https://github.com/huggingface/transformers/tree/067923d3267325f525f4e46f357360c191ba562e) !. 

**Huge thanks to these great open-source projects! This project would be impossible without them.**

## Citation

If you found this code or paper useful, please consider citing:
```
@InProceedings{Duran_2024_WACV,
    author    = {Duran, Enes and Kocabas, Muhammed and Choutas, Vasileios and Fan, Zicong and Black, Michael J.},
    title     = {HMP: Hand Motion Priors for Pose and Shape Estimation From Video},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {6353-6363}
}
```

## Contact
Should you run into any problems or have questions, please create an issue or contact `enes.duran@tuebingen.mpg.de`.

<!-- TODO 
1) MMPose tracking code for in the wild video.  (+)
1) MMPose tracking code for in the lab video.   (+)
1) Keyp blending for in the lab                 (+)
1) Key blending for in the wild                 (-) -->
 
[//]: # (AMASS 800K ARCTIC 391 K frames after processing )
[//]: # (ARCTIC has at most 240Kx2=480K raw frames)
[//]: # (GRAB has 1.6Mx2 = 3.2M raw frames)  
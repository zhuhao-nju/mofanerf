# MoFaNeRF:Morphable Facial Neural Radiance Field
### [Project Page](https://neverstopzyy.github.io/mofanerf/) | [Video](https://neverstopzyy.github.io/mofanerf/video/supplement_video_7_audio_1.mp4) | [Paper](https://arxiv.org/abs/2112.02308)  

[comment]: <> (| [Data]&#40;https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Eo9zn4x_xcZKmYHZNjzel7gBdWf_d4m-pISHhPWB-GZBYw?e=Hf4mz7&#41;)

<img src="https://neverstopzyy.github.io/mofanerf/images/fig_title.png" width=1024>

**Any questions or discussions are welcomed!**

----

## Catalog

-----
* [Install](#install)
* [Test our model](#test-our-model)
  * [Fitting](#fitting)
    + [1. Prepare your data first](#1-prepare-your-data-first)
    + [2. Fit to the processed image](#2-fit-to-the-processed-image)
    + [3. Render images of novel views](#3-render-images-of-novel-views)
    + [4. Refine the rendered results](#4-refine-the-rendered-results)
* [Train your own model](#train-your-own-model)
  + [1. data preprocessing](#1-data-preprocessing)
  + [2. training](#2-training)

## Install

------
```
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>
  
  - PyTorch 1.9.0
  - dlib 19.22.1
  - matplotlib
  - numpy
  - imageio
  - imageio-ffmpeg
</details>
  
------
##  Test our model

------
- ### Fitting
<img src="https://neverstopzyy.github.io/mofanerf/images/fig_gen_fit.png" width=1024>

#### 1. Prepare your data first
We preprocess wild images using the following three steps: alignment, segmentation and relighting, 
and we pack these steps into `wildData_preprocess.py`. After the preprocessing, 
one aligned image and one estimated camera pose you will get.

This demo show how to batch process the wild images.
```
 cd fittingDataPreprocess
 python wildData_preprocess.py --filePath ./data/fit/
```
#### 2. Fit to the processed image
This demo show how to fit our model to target image.
```
python run_fit.py --filePath ./data/fit/segRelRes/00133.png
```
* Reminder: If your GPU has memory of only 10GB, we recommend that you modify parameters "netchunk","chunk" to 16384, "N_rand" to 64 in `./configs/exp_mofanerf.txt`.
#### 3. Render images of novel views
This demo show how to render novel views with the fitted parameters, and just simply add one parameter "--renderType rendering".
```
python run_fit.py --renderType rendering  --filePath ./data/fit/segRelRes/00133.png
```
#### 4. Refine the rendered results
This demo show how to use our refinement module to enrich details of the rendered results.
```
cd trainingDataPreprocess
python test.py --name facescape --nerf_folder ../data/fit/fitting/segRelRes_00133/render/
```


## Train your own model

------

### 1. Data preparation

Data preprocessing contains two parts : 

1. Align all TU models, and then remove the area of shoulder.
2. Render multi-view images.

#### Data download
You should download the facescape dataset from [facescape website]("https://facescape.nju.edu.cn/").
Please place the downloaded dataset in `./data/models_raw/` according to the following direction structure:

```
├── data
│   ├── models_raw
│   │   └── 1
│   │       └── 1_neutral
│   │           ├── 1_neutral.jpg
│   │           ├── 1_neutral.mtl
│   │           ├── 1_neutral.obj
...
```

#### Align and clip

Execute the command below, the aligned and clipped model will be saved in `./data/models_out` folder.

```python
python align_clip.py
```
#### Render

Run the command below to render multi-view images
```
python render.py
```

the results will be saved in `./data/multiViewImages` folder, the structure is 

```
├── data
│   └── multiViewImages
│       └── 1
│           └── neutral
│               ├── -30_0.png
│               └── -30_1.png
....
```
### 2. Training
#### Train MoFaNeRF-coarse
After we set training parameters in `./configs/exp_mofanerf.txt`,
we can directly train your model with this commend. 
```
python run_train.py
```
* Reminder: If you already download our models, you should change parameter 
`expname` in `exp_mofanerf.txt`to 
initial another experiment.

#### Train refinement module
To train the RefineNet,  you should render images with the trained MoFaNeRF-coarse. 
The paired rendered image and its ground truth are placed in `train_A` and `train_B` respectively.


```
├── dataset
│  └── train_A
│  |   └── id
│  |        └── expression
│  |         	└── 1.png
│  |         	└── 2.png
│  └── train_B
│      └── id
│           └── expression
│           	└── 1.png
│           	└── 2.png
...
```


To train the `RefineNet`, run the command below.

```
python train.py --name facescape --dataroot dataset --nerf_folder train_A --gt_folder train_B
```




# Learning Visual Styles from Audio-Visual Associations

###  [Video](https://youtu.be/dskiUJuW-h4) | [Website](https://tinglok.netlify.app/files/avstyle) | [Paper](https://arxiv.org/abs/2205.05072)

<br>

<img src="figs/gif_avstyle.gif" align="center" width=800>

<br><br><br><br>



This repository contains the official codebase for [Learning Visual Styles from Audio-Visual Associations](https://arxiv.org/abs/2205.05072). We manipulate the style of an image to match a sound. After training with an unlabeled dataset of egocentric hiking videos, our model learns visual styles for a variety of ambient sounds, such as light and heavy rain, as well as physical interactions, such as footsteps. We thank Taesung and Junyan for sharing codes of [CUT](https://github.com/taesungp/contrastive-unpaired-translation).



[Learning Visual Styles from Audio-Visual Associations](http://tinglok.netlify.app/files/avstyle)  
[Tingle Li](https://tinglok.netlify.app/), [Yichen Liu](https://www.linkedin.com/in/yichen-liu-751804176/), [Andrew Owens](https://andrewowens.com/), [Hang Zhao](https://hangzhaomit.github.io/)<br>
Tsinghua University, University of Michigan and Shanghai Qi Zhi Institute<br>
In ECCV 2022

## Prerequisites

- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Quick Start

- Clone this repo:

  ```bash
  git clone https://github.com/Tinglok/avstyle avstyle
  cd avstyle
  ```

- Install PyTorch 1.7.1 and other dependencies.

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yaml`.

## Datasets

### Into the wild

We provide Youtube ID in `dataset\Into-the-Wild\metadata.xlsx`. Please see [youtube-dl](https://github.com/ytdl-org/youtube-dl) to download the videos to `dataset/Into-the-Wild/youtube` first. 

Then process them using:
```bash
python dataset\Into-the-Wild\split.py
```

so that the videos are split into 3s video clips.

Then run the command:

```bash
python dataset\Into-the-Wild\video2jpg.py
```

to extract the corresponding images.

Finally download [trainA](https://drive.google.com/file/d/1KSWhf1uVteKqtAS-2XcyA1NzEYekuCtK/view?usp=sharing) and [trainB](https://drive.google.com/file/d/1reWRstlRkXtEPP1AUFuj9T2vXCl_A6yL/view?usp=sharing) to `dataset\Into-the-Wild`.

### The Greatest Hits

Please follow the instruction from [Visually Indicated Sounds](https://andrewowens.com/vis/) to download this dataset.

## Training and Test

- Train our model on the Into the Wild dataset:
```bash
python train.py --dataroot ./datasets/Into-the-Wild --name hiking
```
The checkpoints will be stored at `./checkpoints/hiking/`.

- Train our model on the Greatest Hits dataset:
```bash
python train.py --dataroot ./datasets/Greatest-Hits --name material
```
The checkpoints will be stored at `./checkpoints/material/`.

- Test our model on the Into the Wild dataset:
```bash
python test.py --dataroot ./datasets/Into-the-Wild --eval
```
The test results will be saved to a html file at `./results/hiking/latest_train/index.html`.

- Test our model on the Greatest Hits dataset:
```bash
python test.py --dataroot ./datasets/Greatest-Hits --eval
```
The test results will be saved to a html file at `./results/material/latest_train/index.html`.

## Citation

If you use this code for your research, please consider citing our [paper](https://arxiv.org/abs/2205.05072).

```bash
@inproceedings{li2021learning,
  author={Tingle Li and Yichen Liu and Andrew Owens and Hang Zhao},
  title={{Learning Visual Styles from Audio-Visual Associations}},
  year=2022,
  booktitle={European Conference on Computer Vision (ECCV)}
}
```

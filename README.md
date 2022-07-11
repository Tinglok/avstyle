

# Learning Visual Styles from Audio-Visual Associations

###  [Video](https://youtu.be/dskiUJuW-h4) | [Website](https://tinglok.netlify.app/files/avstyle) | [Paper](https://arxiv.org/abs/2205.05072)

<br>

<img src="figs/gif_avstyle.gif" align="right" width=960>

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

### Quick Start

- Clone this repo:

  ```bash
  git clone https://github.com/Tinglok/avstyle avstyle
  cd avstyle
  ```

- Install PyTorch 1.7.1 and other dependencies.

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yaml`.

### Dataset

#### Into the wild

We provide Youtube ID in `dataset\Into-the-Wild\metadata.csv`. Please follow [youtube-dl](https://github.com/ytdl-org/youtube-dl) to download the videos and process them with `dataset\Into-the-Wild\process.py`.

#### The Greatest Hits

It can be downloaded from [Visually Indicated Sounds](https://andrewowens.com/vis/).

### Training and Test

- Train our model on the Into the Wild dataset:
```bash
python train.py --dataroot ./datasets/hiking --name hiking
```
The checkpoints will be stored at `./checkpoints/hiking/`.

- Train our model on the Greatest Hits dataset:
```bash
python train.py --dataroot ./datasets/material --name material
```
The checkpoints will be stored at `./checkpoints/material/`.

- Test our model on the Into the Wild dataset:
```bash
python test.py --dataroot ./datasets/hiking --eval
```
The test results will be saved to a html file here: ./results/hiking/latest_train/index.html.

- Test our model on the Greatest Hits dataset:
```bash
python test.py --dataroot ./datasets/material --eval
```
The test results will be saved to a html file here: ./results/material/latest_train/index.html.

### Citation

If you use this code for your research, please consider citing our [paper](https://arxiv.org/abs/2205.05072).

```bash
@inproceedings{li2022learning,
  author={Tingle Li and Yichen Liu and Andrew Owens and Hang Zhao},
  title={{Learning Visual Styles from Audio-Visual Associations}},
  booktitle={European Conference on Computer Vision (ECCV)},
  year=2022,
}
```

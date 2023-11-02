# ReLeaPS : Reinforcement Learning-based Illumination Planning for Generalized Photometric Stereo
[\[GitHub\]](https://github.com/jhchan0805/ReLeaPS)
[\[Homepage\]](https://jhchan0805.github.io/ReLeaPS)
[\[Video\]](https://youtu.be/5D4NBlf-L3w)

### Prerequisites
  * The code is tested on Linux with Python 3.10.
  * Install requirements from `requirements.txt` before running.

### Setup
This repo contains sub-module. Clone this repo:
```
git clone --recurse-submodules https://github.com/jhchan0805/ReLeaPS
```
  
  * To train and evaluate on `CNN-PS` backbone, download the pre-trained model `weight_and_model.hdf5` to `data` according to [CNN-PS][1].
  * To train and evaluate on `PS-FCN` backbone, download the pre-trained model `PS-FCN_B_S_32_normalize.pth.tar` to `src/PS-FCN/data/models` according to [PS-FCN][2].
  * To evaluate on `DiLiGenT` dataset, download `DiLiGenT.zip` to `data` according to [DiLiGenT][3].
  
[1]: https://github.com/satoshi-ikehata/CNN-PS-ECCV2018
[2]: https://github.com/guanyingc/PS-FCN
[3]: https://sites.google.com/site/photometricstereodata/single

### Training
  * Download the synthetic dataset for training from: [datasets](https://drive.google.com/file/d/1hZtjtY8DMOk-sITT_AoZzBs5oZzVdgkk/view?usp=drive_link) and place under `data`.
  * Run `run_train.sh`.

### Evaluation
  * Train the models yourself or download the pre-trained models from: [TBD]() and place under `data/models`.
  * Run `run_benchmark.sh`.

## Citation
If you find our work useful for your research, please consider citing:
```BibTeX
@InProceedings{jh2023releaps,
    author = {Chan, Junhoong and Yu, Bohan and Guo, Heng and Ren, Jieji and Lu, Zongqing and Shi, Boxin},
    title = {{ReLeaPS}: Reinforcement Learning-based Illumination Planning for Generalized Photometric Stereo},
    booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2023},
}
```

Copyright (c) 2022-2023 Bohan Yu. All rights reserved. \
ReLeaPS is free software licensed under GNU Affero General Public License version 3 or latter.


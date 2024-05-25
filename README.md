# [EVASR: Edge-Based Video Delivery with Salience-Aware Super-Resolution](https://dl.acm.org/doi/abs/10.1145/3587819.3590967)



### Overview

With the rapid growth of video content consumption, it is important to deliver high-quality streaming videos to users even under limited available network bandwidth. In this paper, we propose EVASR, a system that performs edge-based video delivery to clients with salience-aware super-resolution. We select patches with higher saliency score to perform super-resolution while applying the simple yet efficient bicubic interpolation for the remaining patches in the same video frame. To efficiently use the computation resources available at the edge server, we introduce a new metric called "saliency visual quality" and formulate patch selection as an optimization problem to achieve the best performance when an edge server is serving multiple users. We implement EVASR based on the FFmpeg framework and conduct extensive experiments for evaluation. Results show that EVASR outperforms baseline approaches in both resource efficiency and visual quality metrics including PSNR, saliency visual quality (SVQ), and VMAF.

### prerequisite

Pytorch   
Data can be downloaded here: https://drive.google.com/drive/folders/1dLeoTRousJ6fYE2VpUDdIsZcrzDM8Sf4?usp=share_link

### Saliency Detection

1. The repo for saliency detection https://github.com/guotaowang/STVS   
Download the zip file via https://drive.google.com/file/d/1ZHgdRs-yUzreSFOThOP00MIUVxeRoMYc/view?pli=1

2. unzip the file and go to the directory ``STVS-master``  
modify the model_path ``ckpt_path`` to your model path in line 26 in demo.py  
modify the input frame path ``image_root`` to your input frame path from line 59 to line 64 in demo.py   
modify the save path ``save_name`` to your results directory in line 79 in demo.py  

modify file path split symbol in line 72 in dataset.py
modify the image file type in line 24 and line 40 in dataset.py

Note the input should be at least 3 frames in a video.  

3. run ``python demo.py``

4. modify the ``src_path`` and the ``des_path `` in line 20 and 21 in get_saliency_weight.py   
This script will generate three kinds of weights based on saliency score in each patch.  
Naive weight would be m1   



### Installation
Refer to https://github.com/symmru/FFmpegSR-ISM2022

## Citations

If you find our code or paper helps, please consider citing:
```sh
@inproceedings{li2023evasr,
  title={EVASR: Edge-Based Video Delivery with Salience-Aware Super-Resolution},
  author={Li, Na and Liu, Yao},
  booktitle={Proceedings of the 14th Conference on ACM Multimedia Systems},
  pages={142--152},
  year={2023}
}
```



# EVASR: Edge-Based Video Delivery with Salience-Aware Super-Resolution



### Overview

### prerequist

Pytorch

Data can be download here: https://drive.google.com/drive/folders/1dLeoTRousJ6fYE2VpUDdIsZcrzDM8Sf4?usp=share_link

### Saliency Detection

1. The repo for saliency detection https://github.com/guotaowang/STVS
Download the zip file via https://drive.google.com/file/d/1ZHgdRs-yUzreSFOThOP00MIUVxeRoMYc/view?pli=1

2. unzip the file and go to the directory ``STVS-master``
modify the model_path ``ckpt_path`` to your model path in line 26 in demo.py
modify the input frame path ``image_root`` to your input frame path from line 59 to line 64 in demo.py
modify the save path ``save_name`` to your results directory in line 79 in demo.py

Note the input should be at least 3 frames in a video.

3. run ``python demo.py``

4. modify the ``src_path`` and the ``des_path `` in line 20 and 21 in get_saliency_weight.py
This script will generate three kinds of weights based on saliency score in each patch.
Naive weight would be m1



### Installation
1. clone this repo
```sh
git clone https://github.com/symmru/FFmpegSR-ISM2022.git
```
2. Modify the path to libtorch in _CMakeList.txt_ and typt ``cmake .``with CMakeList
3. Then do ``make`` with Makefile (generated by cmakelist not original makefile in repo. It will generate ``libedsr.so``)
4. Replace the home directory in ``install.sh`` and run ``sudo install.sh`` 
5. Clone the ffmpeg rep
```sh
git clone https://github.com/FFmpeg/FFmpeg
```
6. Copy the ``vf_edsr.c`` to ``FFmpeg/libavfilter`` directory
7. Register the filter with FFmpeg. Add ``extern AVFilter ff_vf_edsr`` to ``libavfilter/allfilters.c``
8. Add ``OBJS-$(CONFIG_EDSR_FILTER) += vf_edsr.o`` to ``libavfilter/Makefile``
9. In the root FFmpeg directory, configure FFmpeg:
```sh
./configure --enable-gpl --enable-nonfree --enable-libass --enable-libfdk-aac --enable-libfreetype --enable-libvpx --enable-libx264 --enable-libxvid --extra-libs='-lstdc++ -ledsr'
```
10. Directly add lib torch PATH into ``~./bashrc``:
```sh
export LD_LIBRARY_PATH=$LD_LIBRART_PATH:/home/lina/libtorch/lib
export PATH=$PATH:/home/lina/libtorch/bin'\
```
11. make FFmpeg with command ``make``
12. Test with command
```sh
./ffmpeg -benchmark -i path_to_input -filter_complex '
[0:v] format=pix_fmts=yuv420p, extractplanes=y+u+v [y][u][v];
[y] edsr=width=960:height=540:patch_width=160:patch_height=90:scale=2:file_path=path_to_patch_file:model_path=path_to_model [y_scaled];
[u] scale=iw*2:ih*2 [u_scaled];
[v] scale=iw*2:ih*2 [v_scaled];
[y_scaled][u_scaled][v_scaled] mergeplanes=0x001020:yuv420p [merged]
' -map [merged] -c:v libx264 -preset ultrafast -crf 0 path_to_output
```

In ``edsr`` filter, **width** and **height** stands for the resolution of input video frame. **patch_with** and **patch_height** stands for the resolution of patch frame. Our default setting is 6x6 grid per frame.  **file_path** stands for the patch information that need to apply super resolution. **model_path** is the traced model with ``torchscript``.


## License

MIT



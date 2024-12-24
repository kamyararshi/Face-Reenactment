# Unsupervised Face Reenactment
Unofficial pytorch implementation of paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing".  

```Python 3.6``` and ```Pytorch 1.7``` are used. 

To Do's:
--------
1. Change the equivariance keypoints loss from calculating on 2D to 3D keypoints
2. Predict flow masks using a self-attention mechanism or transformer enc-decoder
3. Estimate the warping fields in a better way than Fist-Order approx

Train:  
--------
```
python run.py --config config/vox-256.yaml --device_ids 0,1,2,3
```

Demo:  
--------
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame
```
free-view (e.g. yaw=20, pitch=roll=0):
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame --free_view --yaw 20 --pitch 0 --roll 0
```
Note: run ```crop-video.py --inp driving_video.mp4``` first to get the cropping suggestion and crop the raw video.  

Pretrained Model:  
--------

  Model  |  Train Set   | - | - | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | - |  -  |

 
 Note:
 1. <s>For now, the Beta Version is not well tuned.</s>
 2. For free-view synthesis, it is recommended that Yaw, Pitch and Roll are within ±45°, ±20° and ±20° respectively.
 3. Face Restoration algorithms ([GPEN](https://github.com/yangxy/GPEN)) can be used for post-processing to significantly improve the resolution.
![show](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/s%20r.gif) 


Acknowlegement: 
--------
Thanks to [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model), [Wang et. al](https://arxiv.org/pdf/2011.15126), and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose).

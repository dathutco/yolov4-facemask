# yolov4-facemask
I will build model to api, it isn't complete.

I still test by using Flask, so I won't upload new source now
Any thing you have to do: Extract need.zip and copy all to the folder(gpu/cpu) that you can run


# How to run: on ./gpu or on ./cpu

## With cpu
#for image

darknet_no_gpu.exe detector test data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../image/demo.png -thresh 0.3 -dont_show

#for video

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../video/video.mp4 -thresh 0.8 -dont_show -out_filename ../video/output.mp4

#camera

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights

## With gpu

You just replace darknet_no_gpu.exe to darknet.exe

# Requirement
With GPU, you need many app relate to darknet: 
* Visual Studio(I use 2019)
* CUDA vs CUDNN for using GPU (I use 11.4 vs 8.2.4)
* OPENCV (I use 3.4)

With CPU, you just install Visual studio vs Opencv

# Notice

if you want to run on notebook, replace "a.exe" by !./a
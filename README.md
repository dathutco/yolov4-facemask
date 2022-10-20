# yolov4-facemask
I will build model to api, it isn't complete.

I still test by using Flask, so I won't upload new source now
# Requirement

* CUDA vs CUDNN for using GPU (I use 11.4 vs 8.2.4)
# Notice

Any thing you have to do: Extract all file need.rar in need folder and copy all from that to the folder(gpu/cpu) that you can run.
# How to run: on GPU or on CPU

## With CPU
#for image

darknet_no_gpu.exe detector test data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../image/demo.png

#for video

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../video/video.mp4 -out_filename ../video/output.mp4

#camera

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights

## With GPU

You just replace darknet_no_gpu.exe to darknet.exe
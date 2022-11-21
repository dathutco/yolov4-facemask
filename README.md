# yolov4-facemask
I will build model to api.

The Web allow user validate the accuracy of predicting label. Now, I just save image and file label (label, x,y,w,h) when the confidence >0.9.

I save on local and Firebase (something wrong from Firebase so I push all to both, still tuning)

This project also have Data Visualization.

# Requirement

* CUDA vs CUDNN for using GPU (I use 11.4 vs 8.2.4)
# Notice

Any thing you have to do: Extract all file need.rar in need folder and copy all from that to the folder(gpu/cpu) that you can run.
# How to run: on GPU or on CPU

## Two way to run: use darknet and use build API

# +For Darknet

## With CPU
#for image

darknet_no_gpu.exe detector test data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../image/demo.png

#for video

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights ../video/video.mp4 -out_filename ../video/output.mp4

#camera

darknet_no_gpu.exe detector demo data/obj.data cfg/yolov4.cfg Model/yolov4-custom_best.weights

## With GPU

You just replace darknet_no_gpu.exe to darknet.exe

# -For API

On gpu dir, you can see $api.py$, $backup.py$, $client.py$ vs $newway.py$

* $backup.py$: for build a api, but it slow cause I write for cpu computer
* $client.py$: file from user that want to use api
* $app.py: run on web

* For $new.py$: this is improvement for api.py file, but it has some proglem with box
# yolov4-facemask
I will build model to api.

I still test by using Flask, so I won't upload new source now
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

## If you don't have GPU, run "backup" file, else see how to active opencv using GPU, run "new" file.

* $backup.py$: for build a api, but it slow cause I write for cpu computer
* $api.py$: as same as backup, but this can run with gpu, but you must install open-cv with a special way.
* $client.py$: file from user that want to use api
* For $new.py$: this is improvement for api.py file.

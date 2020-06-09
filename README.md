## Introduction
CheetahInfer is a pure C++ inference SDK based on TensorRT, which supports fast inference of classifier and object detector.

## Prerequisites
CheetahInfer has several dependencies:
- OpenCV
  - tested on version 4.3.0
- CUDA
  - tested on version 10.1
- TensorRT
  - tested on version 6.0.1.5
- cuDNN
  - optional
  - tested on version 7.6.5
- GCC
  - tested on version 5.4.0
After the installation of above dependencies, we need modify the `TENSORRT_INSTALL_DIR` and `OPENCV_INSTALL_DIR` in `Makefile.config` accordingly.

## Preparation for model and data
- Prepare the ONNX file
  - If your model has a PyTorch format, you can use [vedadep](https://github.com/Media-Smart/volksdep) to convert PyTorch model to ONNX model.
- Modify the ONNX file path and some related configurations in `main.cpp` in `classifier` or `retinanet` folder
- Get some images for testing

## Compilation and running
```
cd classifier
make -j12
./build/main --imgfp /path/to/image
```
or
```
cd classifier
make -j12
./build/main --imgfp /path/to/image
```
If you want speficy which GPU to use, you can make it by setting the environment variable `CUDA_VISIBLE_DEVICES`.

## Credits
We got a lot of code from [TensorRT](https://github.com/NVIDIA/TensorRT) and [retinanet-examples](https://github.com/NVIDIA/retinanet-examples).

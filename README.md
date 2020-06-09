## Introduction
CheetahInfer is a C++ inference SDK based on TensorRT.

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
- gcc
  - tested on version 5.4.0
After the installation of dependencies, we should modify the `TENSORRT_INSTALL_DIR` and `OPENCV_INSTALL_DIR` accordingly.

## Preparation for model and data
- Prepare the ONNX file
- Modify the ONNX file path and some related configuration in `main.cpp` in `classifier` or `retinanet` folder
- Get some images

## Compilation and running
```
cd classifier
make
./build/main --imgfp /path/to/image
```
or
```
cd classifier
make
./build/main --imgfp /path/to/image
```
## Credits
We got a lot of code from [TensorRT](https://github.com/NVIDIA/TensorRT) and [retinanet-examples](https://github.com/NVIDIA/retinanet-examples).


<div align="center">

# TwinLiteNet ONNX Model Inference with ONNX Runtime

</div>

This repository includes a C++ implementation for performing inference with the state-of-the-art [TwinLiteNet model](https://github.com/chequanghuy/TwinLiteNet) using ONNX Runtime. TwinLiteNet is a cutting-edge lane detection and drivable area segmentation model. This implementation provides support for both CUDA and CPU inference through build options.



<div align="center">
    <table>
        <tr>
            <td><img src="assets/results0.jpg" alt="Image 1" width="300"/></td>
            <td><img src="assets/results2.jpg" alt="Image 2" width="300"/></td>
            <td><img src="assets/results3.jpg" alt="Image 3" width="300"/></td>
        </tr>
        <tr>
            <td><img src="assets/results4.jpg" alt="Image 4" width="300"/></td>
            <td><img src="assets/results5.jpg" alt="Image 5" width="300"/></td>
            <td><img src="assets/results6.jpg" alt="Image 6" width="300"/></td>
        </tr>
    </table>
</div>
<br>

## Acknowledgment 🌟

I would like to express sincere gratitude to the creators of the [TwinLiteNet model](https://github.com/chequanghuy/TwinLiteNet) for their remarkable work .Their open-source contribution has had a profound impact on the community and has paved the way for numerous applications in autonomous driving, robotics, and beyond.Thank you for your exceptional work.
<br>
<br>

## Project Structure

The project has the following structure:

```

├── CMakeLists.txt
├── LICENSE
├── README.md
├── assets/
├── images/
├── include/
│   └── twinlitenet_onnxruntime.hpp
├── models/
│   └── best.onnx
└── src/
    ├── main.cpp
    └── twinlitenet_onnxruntime.cpp
```
## Requirements

- [ONNX Runtime](https://onnxruntime.ai/)
- OpenCV
<br>

## Build Options

- **CUDA Inference:** To enable CUDA support for GPU acceleration, build with the `-DENABLE_CUDA=ON` CMake option.
- **CPU Inference:** For CPU-based inference, no additional options are required.
<br>

## Usage

1. Clone this repository.
2. Build the project using CMake with your preferred build options.
```cpp
mkdir build
cd build
cmake  -DENABLE_CUDA=ON ..
make -j8
```
4. Execute `./main` and Enjoy accurate lane detection and drivable area results!
<br>

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use it in both open-source and commercial applications.
<br><br>

## Extras

- [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) 
- [TwinLiteNet-OpenCV-DNN](https://github.com/harrylal/TwinLiteNet-onnx-opencv-dnn)

[![CI](https://github.com/embedded-dev-research/ITLabAI/actions/workflows/ci.yml/badge.svg)](https://github.com/embedded-dev-research/ITLabAI/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/embedded-dev-research/ITLabAI/graph/badge.svg?token=L3OS8C4BI6)](https://codecov.io/gh/embedded-dev-research/ITLabAI)

# AlexNet-MNIST-Inference
## Model Performance

<!--ACCURACY_PLACEHOLDER-->Accuracy: Stat: 98.01% (updated: 2025-04-28)<!--END_ACCURACY-->
## Short description
A lightweight C++ library for performing high-performance inference on classification tasks. Designed for efficiency and educational purposes, this project demonstrates how classic CNNs can be optimized for small-scale tasks in native environments.
### Key Features:

* C++17 implementation for bare-metal performance

* Simplified AlexNet for 28×28 grayscale images

* Googlenet, Densenet, Resnet and Yolo11x-cls for images of any size

* Parallel computing via Intel OneTBB (Threading Building Blocks)

* Pre-trained model: AlexNet-model.h5, Googlenet.onnx included
## **Some files used to create the library**
### Neural network models
You need to download [Alexnet-model.h5](https://github.com/moizahmed97/Convolutional-Neural-Net-Designer/blob/master/AlexNet-model.h5) to the folder *docs*

Other models:</br>
[GoogLeNet.onnx](https://huggingface.co/qualcomm/GoogLeNet/resolve/main/GoogLeNet.onnx)</br>
[yolo11x-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-cls.pt)</br>
[resnest101e_Opset16.onnx](https://github.com/onnx/models/raw/refs/heads/main/Computer_Vision/resnest101e_Opset16_timm/resnest101e_Opset16.onnx?download=)</br>
[densenet121_Opset16.onnx](https://github.com/onnx/models/raw/refs/heads/main/Computer_Vision/densenet121_Opset16_timm/densenet121_Opset16.onnx?download=)



## **How do I launch the inference?**
* Make sure you install the project dependencies by running: *pip install -r requirements.txt*
* You need to run the script *parser.py* that is located in app/converters to read weights from a model *Alexnet-model.h5* or *parser_onnx.py* to read weights from a models ONNX or YOLO and the json file with the weights will be stored in the *docs* folder.
* Then put the test images in png format in the folder *docs/input*
* After building the project, which is described below, run Graph_build with the parameter --model (alexnet_mnist or googlenet or densenet or resnet or yolo) and the parameter --parallel if you need. App Graph_build is located in folder *build/bin*

## **Building a Project**
### *Windows*
To build and run this project locally on Windows, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/ITLabAI.git
   ```
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Configure the project:**
   Create a separate directory for configure the project and compile it:
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
    ```
   If you want to build in a Debug, change the Release to Debug

   *Note: Make sure you have CMake installed to build the project.*
5. **Build the project:**
   Next, to build the project, we will need to enter the command
    ```bash
   cmake --build . --config Release
    ```
6. **Run the project**
   After building the project, you can find the executable file in the following path from the *build* folder
   ```bash
   cd build/bin
    ```
   and run the file
    ```bash
   Graph_Build.exe --model alexnet_mnist
    ```
### *Linux/macOS*
   To build and run this project locally on Linux or macOS, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/ITLabAI.git
   ```
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Install necessary dependencies:**
   1. OpenMP
     Debian/Ubuntu:
     ```bash
     sudo apt-get install -y libomp-dev
     ```
     macOS:
     ```
     brew install libomp
     ```
4. **Configure the project:**
   Create a separate directory for configure the project and compile it:
   ```bash
   cmake -S . -B build
    ```
    *Note: Make sure you have CMake installed to build the project.*
   For macOS need to specify the path to omp.h file:
   ```bash
   cmake -S . -B build -DCMAKE_CXX_FLAGS="-I$(brew --prefix libomp)/include" -DCMAKE_C_FLAGS="-I$(brew --prefix libomp)/include"
   ```
5. **Build the project:**
   Next, to build the project, we will need to enter the command
    ```bash
   cmake --build build --config Release
    ```
    If you want to build in a Debug, change the Release to Debug
6. **Run the project**
   After building the project, you can find the executable file in the following path from the *build* folder
   ```bash
   cd build/bin
    ```
   and run the file
    ```bash
   ./Graph_Build --model alexnet_mnist
    ```

## Test Process
   This project contains tests to verify functionality.
   To test the project, the Google Test Framework is used as a submodule of the project.
   ### Google Test Framework

   Google Test is a powerful framework for unit testing in C++. In this project, Google Test is a submodule. When building the project, you have already       updated it, and it is ready for use.
   ### Running tests
   ### *Windows*
   
   To start the testing process locally, you need to go to the directory
   ```bash
   cd build/bin
   ```
   and run the following files:
   ```bash
   run_test.exe
   ```
### *Linux*
To start the testing process locally, you need to go to the directory
   ```bash
   cd build/bin
   ```
   and run the following files:
   ```bash
   chmod +x run_test
   ./run_test
   ```

## **Accuracy validation for Alexnet on MNIST**
To run accuracy validation you need to use the MNIST dataset, which you can download [here](https://github.com/DeepTrackAI/MNIST_dataset/tree/main/mnist/test) and put it in a folder *docs/mnist/mnist/test*
Now you can run accuracy check - *build\bin\ACC.exe --model alexnet_mnist*
* **The accuracy should be 98.01%**

## **Accuracy validation for ONNX or YOLO models on ImageNet**
To run accuracy validation you need to use the ImageNet dataset, which you can download [here](https://www.kaggle.com/datasets/sautkin/imagenet1kvalid) and put it in a folder *docs/Imagenet/*
Now you can run accuracy check - *build\bin\ACC.exe --model googlenet*

## **Documentation of project**
https://github.com/embedded-dev-research/ITLabAI/blob/Semyon1104/Final_documentation/docs/IT_Lab_2023.pdf
## **Structure of our library**
![Class diagram](./docs/class_diagram.svg)

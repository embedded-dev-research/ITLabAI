[![CI](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml)

# itlab_2023
# Name
## Short description ##
# **Building a Project**
To build and run this project locally on Windows, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/itlab_2023.git
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Configure the project:**
   Create a separate directory for configure the project and compile it:
   ```bash
   mkdir build
   cd build
   cmake ..
    ```
    *Note: Make sure you have CMake installed to build the project.*
4. **Build the project:**\
   Next, to build the project, we will need to enter the command
    ```bash
   cmake --build . --config Debug
    ```
    If you want to build in a release, change the debug to release\
5. **Run the project**
   After building the project, you can find the executable file in the following path from the *build* folder
   ```bash
   cd bin\Debug
    ```
   and run the file
    ```bash
   Reader.exe
    ```
# Test Process
   This project contains tests to verify functionality.
   To test the project, the Google Test Framework is used as a submodule of the project.
   ### Google Test Framework

   Google Test is a powerful framework for unit testing in C++. In this project, Google Test is a submodule. When building the project, you have already       updated it, and it is ready for use.
   ### Running tests
   To start the testing process locally, you need to go to the directory
   ```bash
   cd bin\Debug
   ```
   and run the following files:
   ```bash
   run_test.exe
   run_test_read.exe
   ```

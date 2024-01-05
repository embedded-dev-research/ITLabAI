[![CI](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml/badge.svg)](https://github.com/embedded-dev-research/itlab_2023/actions/workflows/ci.yml)

# itlab_2023
# Name
## Short description ##
# **Building a Project**
To build and run this project locally, follow these steps:

1. **Clone the repository:**  
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/embedded-dev-research/itlab_2023.git
2. **Update submodules:**
   Navigate to the project directory and update the submodules:
   ```bash
   git submodule update --init --recursive
3. **Build the project:**
   Create a separate directory for building the project and compile it:
   ```bash
   mkdir build
   cd build
   cmake ..
    ```
4. **Open in your code editor:**
   Once the project is successfully built, open it in your preferred code editor.\
   *Note: Make sure you have CMake installed to build the project.*
# Test Process
   This project contains tests to verify functionality.
   To test the project, the Google Test Framework is used as a submodule of the project.
   ### Google Test Framework

   Google Test is a powerful framework for unit testing in C++. In this project, Google Test is a submodule. When building the project, you have already       updated it, and it is ready for use.
   ### Running tests

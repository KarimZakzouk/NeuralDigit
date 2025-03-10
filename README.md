# NeuralDigit - NeuralDigit Neural Network Digit Recognition  (C++)

This C++ project implements a neural network for handwritten digit recognition using the MNIST dataset. The network architecture consists of three layers with 785, 30, and 10 neurons respectively.

## Image Preprocessing (Web Version)

The process begins by scaling down the bounding box of the input drawing to a size of 20x20 pixels. The image is then centered on the center of mass of the pixels within a 28x28 image. This preprocessing step prepares the image for feeding it into the pre-trained neural network.

## Pre-Trained Model (Web Version)

The neural network model used in this project is pre-trained using the MNIST dataset, following the same instructions as the MNIST training data. The model has learned to recognize handwritten digits based on this training.

## Usage

### Running the Neural Network

1. Clone this repository and navigate to the project directory.
2. Extract the `Data.zip` file located in the `data` directory.
3. Compile the project:
   ```sh
   g++  NeuralDigit.cpp -o NeuralDigit
   ```
4. Run the neural network:
   ```sh
   ./NeuralDigit
   ```

### Input Data Format

The necessary input files for training and testing are located inside the `data` directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

These files contain the MNIST dataset images and labels.

Feel free to contribute, report issues, or make suggestions for improvements!


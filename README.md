# PestDetective

PestDetective is an open-source project that combines AI and machine learning to provide a comprehensive pest control solution. The project includes three key features: uploading pest photos for detection, using an ESP32 for pest detection, and training a Teachable Machine model to classify pest images.
We're using Flask and AWS EC2 instance to deploy this app. 

## Features
- Pest photo detection: Users can upload a pest photo, and the YOLOv3 model will detect the type of pest and display bounding boxes around it.
- ESP32 pest detection: Users can use an ESP32 board with a PIR sensor to detect pest movement, and the YOLOv3 model will detect the type of pest and display bounding boxes around it.
- Teachable Machine: Users can upload a zip folder of images and train a custom TensorFlow/Keras model to classify pests. The trained model can then be used to classify new images.

## Installation
To use PestDetective, you will need to install the required dependencies. Refer to requirements.txt.

Copy the requirements.txt file from the repo into your working directory and install those requirements explicitly with
```
$ pip install -r requirements.txt
```
Install tflite-model-maker with
```
$ pip install tflite-model-maker
```
There seems to be a bug right now where tflite-model-maker is not aware of its own requirements (or something funky with pip is happening) and it is trying to resolve them algorithmically instead of just reading the requirements file.

## Usage
To use PestDetective, you can follow the detailed usage instructions in the usage guide. This guide provides step-by-step instructions on how to use each of the three features.

## Documentation
The documentation directory contains detailed documentation for each module and function in the code. This documentation includes descriptions of what each function does, the parameters it takes, and the return values.

## Examples
The examples directory includes sample images and data to demonstrate how each of the three features works.

## License
...

## Credits
PestDetective was created by Manan Luthra. 

## References:

- YOLOv3 model: [link to source]
- TensorFlow/Keras: [link to source]
- ESP32 firmware: [link to source]


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
### Microcontroller used: <br /> LILYGO® TTGO T-Camera ESP32 WROVER & PSRAM (OV2640 Camera Module) <br />

### Pin Configuration for TTGO ESP32:
#define PWDN_GPIO_NUM       -1 <br />
#define RESET_GPIO_NUM      -1 <br />
#define XCLK_GPIO_NUM       4 <br />
#define SIOD_GPIO_NUM       18 <br />
#define SIOC_GPIO_NUM       23 <br />
#define Y9_GPIO_NUM         36 <br />
#define Y8_GPIO_NUM         37 <br />
#define Y7_GPIO_NUM         38 <br />
#define Y6_GPIO_NUM         39 <br />
#define Y5_GPIO_NUM         35 <br />
#define Y4_GPIO_NUM         14 <br />
#define Y3_GPIO_NUM         13 <br />
#define Y2_GPIO_NUM         34 <br />
#define VSYNC_GPIO_NUM      5 <br />
#define HREF_GPIO_NUM       27 <br />
#define PCLK_GPIO_NUM       25 <br />

(Note: The PIR sensor has a deep sleep issue)


### Executing the ESP32 code on Arduino IDE
1. Open the .ino file provided in the repo
2. Make sure connect the correct COM port and Dev module for ESP 
3. Enable PSRAM and set storage to HUGE <br />
![Screenshot 2022-12-06 140623](https://user-images.githubusercontent.com/105019328/206023121-50a0df8c-837b-44f7-a771-cadee9c211df.jpg)

4. Change the SSID and Password in the code <br />
![Screenshot 2022-12-06 140602](https://user-images.githubusercontent.com/105019328/206022962-3b4ec1d2-a76a-4612-83a4-60ca1d272604.jpg)

5. Update server details

6. Compile and upload the code



## Running the flask app on EC2 instance 
1. Launch an EC2 instance on the AWS console and choose an appropriate Amazon Machine Image (AMI) for your instance.
2. During the launch wizard, make sure to configure the security group to allow incoming traffic on the port that your Flask app is listening on (default is 5000).
3. Once the instance is running, copy your Flask app files to the instance using any file transfer method such as SCP or SFTP (I prefer Git pulls)
4. Install the requirements 
5. Run the flask app 
```
python app.py
```

## Credits
PestDetective was developed by Manan Luthra. 

## References:

- <a href="https://pjreddie.com/darknet/yolo/">YOLOv3 model</a> 
- <a href="https://github.com/theAIGuysCode/Object-Detection-API">The AI Guy</a>
- <a href="https://www.tensorflow.org/lite/models/modify/model_maker/image_classification">TensorFlow/Keras</a>
- <a href="https://www.tensorflow.org/lite/models/modify/model_maker/object_detection">Tensorflow Light Model Maker</a>
- <a href="https://storage.googleapis.com/openimages/web/index.html">Google Open Images Dataset</a>
- <a href="https://roboflow.com/convert/oidv4-txt-to-pascal-voc-xml">Roboflow</a>



import time
from absl import app
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort, render_template, send_file, session, flash, url_for, redirect

import os
from werkzeug.utils import secure_filename
import io
from keras import layers

import pathlib
import zipfile
from fileinput import filename
from keras.models import Sequential, load_model
from keras_preprocessing.image import load_img, img_to_array
from PIL import Image
import time
import shutil

from tflite_model_maker import model_spec
from tflite_model_maker import object_detector
from tflite_support import metadata


assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

from PIL import Image
import platform
from typing import List, NamedTuple
import json


# customize your API through the following parameters
classes_path = 'data/labels/obj.names'
weights_path = 'weights/yolov3.tf'
tiny = False                    # set to True if using a Yolov3 Tiny model
size = 416                      # size images are resized to for model
# output_path = 'detections/'   # path to output folder where images with detections are saved
num_classes = 5                # number of classes in model 

# load in weights and classes
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if tiny:
    yolo = YoloV3Tiny(classes=num_classes)
else:
    yolo = YoloV3(classes=num_classes)

yolo.load_weights(weights_path).expect_partial()
print('weights loaded')

class_names = [c.strip() for c in open(classes_path).readlines()]
print('classes loaded')



# Define constants for training app
UPLOAD_FOLDER='static/uploads/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
img_height=180
img_width =180
batch_size=32  #change this to 32 later
IMG_SIZE = (img_height, img_width)


# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"


@app.route('/', methods=["GET",'POST'])
def home():
    return render_template('index.html')

@app.route('/pest_detect_home')
def pest_home_image():
    return render_template('detect_home.html')

# Flask app for upload_image pest detection 
@app.route('/image')
def image():
    return render_template('image.html')

@app.route('/image', methods=['POST'])
def upload_image(): 
    #if(request.method == "POST"):
        # print(request.files["esp32-cam"].read())
    # image=request.form.get('file')
    # image=request.files["file"].read()
    image=request.files['img_file']
    if image:
        # data=[]
        # json_responses=[]
        image_name=secure_filename(image.filename)
        #image = request.files["images"]
        #image_name = image.filename
        #image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
         
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        

        # prepare image for response
        _, img_encoded = cv2.imencode('.png', img)
        response = img_encoded.tobytes()
        img=Image.open(io.BytesIO(response))
        #response = img_encoded.tostring()
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
  
        flash('Image successfully uploaded and displayed below')
        return render_template('image.html', filename=image_name)
        

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/json', methods=["GET",'POST'])
def JSON():
    return render_template('json.html')

# API that returns JSON with classes found in images
@app.route('/json', methods=["GET",'POST'])
def get_detections():
    raw_images = []
    images = request.files.getlist("img_file")
    image_names = []
    for image in images:
        image_name = image.filename
        image_names.append(image_name)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        raw_images.append(img_raw)
        
    num = 0
    
    # create list for final response
    response = []

    for j in range(len(raw_images)):
        # create list of responses for current image
        responses = []
        raw_img = raw_images[j]
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
            responses.append({
                "class": class_names[int(classes[0][i])],
                "confidence": float("{0:.2f}".format(np.array(scores[0][i])*100))
            })
        response.append({
            "image": image_names[j],
            "detections": responses
        })
        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(output_path + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(output_path + 'detection' + str(num) + '.jpg'))

    #remove temporary images
    for name in image_names:
        os.remove(name)
    try:
        return jsonify({"response":response}), 200
    except FileNotFoundError:
        abort(404) 


         

#@app.route('/esp32post')
#def home_new():
#    return render_template('esppostimage.html')

@app.route('/esp32post', methods=['GET','POST'])
def uploadimage32():                 
    image=request.files.get("file", None)
    if image:
        image_name=secure_filename(image.filename)
        image.save(os.path.join(os.getcwd(), image_name))
        img_raw = tf.image.decode_image(
            open(image_name, 'rb').read(), channels=3)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, size)   

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        #cv2.imwrite(output_path + 'detection.jpg', img)
        # print('output saved to: {}'.format(output_path + 'detection.jpg'))
        
        # prepare image for response
        _, img_encoded = cv2.imencode('.jpg', img)
        response = img_encoded.tobytes()
        img=Image.open(io.BytesIO(response))
        #response = img_encoded.tostring()
        print('The image name', image_name)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'esp_image.jpg'))
        img=load_img(os.path.join(app.config['UPLOAD_FOLDER'], 'esp_image.jpg'))
        # image_path=os.path.join(UPLOAD_FOLDER, 'esp_image.png')

    return render_template('esppostimage.html', filename='esp_image.jpg')
    

@app.route('/display_esp/<filename>')
def display_esp(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/teach_mach')
def train_home():
    return render_template('teach_mach.html')

@app.route('/teach_mach', methods=['GET','POST'])
def train():
    uploaded_files = request.files.getlist('train_images')
    filenames = []
    for file in uploaded_files:
        filename = file.filename
        if filename != '':
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # filenames.append(filename)
    
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()
    print('Zip received')    
    
    file_stem=pathlib.Path('static/uploads/'+filename).stem
    data_dir = pathlib.Path('static/uploads/'+file_stem)
    print(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    print('image count complete')
    flash('IMAGE COUNT COMPLETE')
    # flash('INITIALIZING TRAINING')
    # flash('Note: This may take a while! Please wait')

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)
    array=[]
    array=class_names
    flash(array)
    
        # class_names_array = np.array(i, ndmin=2)
    np.savetxt('static/tm_class.txt', array, delimiter=',', newline='\n', fmt='%s')

    print('class names printed')

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    # data_augmentation = Sequential(
    #     [
    #         layers.RandomFlip("horizontal",
    #                         input_shape=(img_height,
    #                                     img_width,
    #                                     3)),
    #         layers.RandomRotation(0.1),
    #         layers.RandomZoom(0.1),
    #     ]
    # )

    model = Sequential([
    #data_augmentation,    
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Dropout(0.2),
    
    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    #layers.Dense(10, activation='softmax'),
    layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    model.summary()
    # flash(model.summary())
    epochs=10
    history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
    )

    flash('Training complete')

    model.save('static/trained_weights.h5')

    weights_path = 'static/trained_weights.h5'


    # send_file(weights_path, as_attachment=True)
    return send_file(weights_path, as_attachment=True)
    # return render_template('teach_mach.html')
   


@app.route('/TM_test')
def test_home():
    return render_template('TM_test.html')

@app.route('/TM_test', methods=['GET','POST'])
def test():
    
    model_upload=request.files["model_file"]
    image=request.files["test_image"]
    filename = model_upload.filename
    model_upload.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # model_upload.save('C:/Users/manan/trained_weights.h5')
    model=load_model(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # model=load_model('C:/Users/manan/trained_weights.h5')

    class_names = np.loadtxt('static/tm_class.txt', delimiter=',', dtype=str, comments=None)

    # Print the class names
    print(class_names)

    
    if image:
        image_name=secure_filename(image.filename)
        #image.save(os.path.join(os.getcwd(), image_name))
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

    img= load_img(os.path.join(app.config['UPLOAD_FOLDER'],image_name), target_size=(img_height, img_width))

    # Convert the image to a NumPy array and normalize its pixel values
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    # img_array /= 255.0 # Normalize pixel values
    
    probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
    probs = probability_model.predict(img_array)

    print(probs[0])
    print(np.argmax(probs[0]))
    

    flash(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(probs)], 100 * np.max(probs))
    )

    return render_template('TM_test.html', filename=image_name)

@app.route('/display_test/<filename>')
def display_test(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/objdet')
def objdet():
    return render_template('objectdetect.html')

@app.route('/objdet', methods=["GET",'POST'])
def bound_box():
    # train_upload=request.files.get('train_images')
    # valid_upload=request.files.get('valid_images')
    uploaded_files = request.files.getlist('train_images')

    class_upload=request.files.get('class_names') # should be a txt file

       
    # for train_file in train_upload:
    #     train_filename=train_file.filename
    #     if train_filename != '':
    #         train_file.save(os.path.join(app.config['UPLOAD_FOLDER'], train_filename))
    
    # zip_train = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, train_filename), 'r')
    # zip_train.extractall(UPLOAD_FOLDER)
    # zip_train.close()
    # print('Train images received')

    # for valid_file in valid_upload:
    #     valid_filename=valid_file.filename
    #     if valid_filename != '':
    #         valid_file.save(os.path.join(app.config['UPLOAD_FOLDER'], valid_filename))
    
    # zip_valid = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, valid_filename), 'r')
    # zip_valid.extractall(UPLOAD_FOLDER)
    # zip_valid.close()
    # print('Validation images received')

    
    for file in uploaded_files:
        filename = file.filename
        if filename != '':
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # filenames.append(filename)
    
            zip_ref = zipfile.ZipFile(os.path.join(UPLOAD_FOLDER, filename), 'r')
            zip_ref.extractall(UPLOAD_FOLDER)
            zip_ref.close()
    print('Zip received') 


    class_names = []
    if class_upload is not None:
        contents = class_upload.read().decode('utf-8')
        class_names = contents.splitlines()

    np.savetxt('static/class.txt', class_names, delimiter=',', newline='\n', fmt='%s')


    spec = model_spec.get('efficientdet_lite0')

    train_dir=os.path.join(UPLOAD_FOLDER, 'train')
    valid_dir=os.path.join(UPLOAD_FOLDER, 'valid')
    print(class_names)

    train_data = object_detector.DataLoader.from_pascal_voc(
        train_dir,
        train_dir,
        class_names
    )

    val_data = object_detector.DataLoader.from_pascal_voc(
        valid_dir,
        valid_dir,
        class_names
    )

    model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, epochs=20, validation_data=val_data)

    model.evaluate(val_data)

    model.export(export_dir='static/')

    model.evaluate_tflite('static/model.tflite', val_data)

    model_path='static/model.tflite'

    return send_file(model_path, as_attachment=True)



@app.route('/objdet_test')
def objdet_test():
    return render_template('objdettest.html')

@app.route('/objdet_test',methods=['GET','POST'])
def boundbox_test():
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate

    # pylint: enable=g-import-not-at-top


    class ObjectDetectorOptions(NamedTuple):
        """A config to initialize an object detector."""

        enable_edgetpu: bool = False
        """Enable the model to run on EdgeTPU."""

        label_allow_list: List[str] = None
        """The optional allow list of labels."""

        label_deny_list: List[str] = None
        """The optional deny list of labels."""

        max_results: int = -1
        """The maximum number of top-scored detection results to return."""

        num_threads: int = 1
        """The number of CPU threads to be used."""

        score_threshold: float = 0.0
        """The score threshold of detection results to return."""


    class Rect(NamedTuple):
        """A rectangle in 2D space."""
        left: float
        top: float
        right: float
        bottom: float


    class Category(NamedTuple):
        """A result of a classification task."""
        label: str
        score: float
        index: int


    class Detection(NamedTuple):
        """A detected object as the result of an ObjectDetector."""
        bounding_box: Rect
        categories: List[Category]


    def edgetpu_lib_name():
        """Returns the library name of EdgeTPU in the current platform."""
        return {
            'Darwin': 'libedgetpu.1.dylib',
            'Linux': 'libedgetpu.so.1',
            'Windows': 'edgetpu.dll',
        }.get(platform.system(), None)


    class ObjectDetector:
        """A wrapper class for a TFLite object detection model."""

        _OUTPUT_LOCATION_NAME = 'location'
        _OUTPUT_CATEGORY_NAME = 'category'
        _OUTPUT_SCORE_NAME = 'score'
        _OUTPUT_NUMBER_NAME = 'number of detections'

        def __init__(
            self,
            model_path: str,
            options: ObjectDetectorOptions = ObjectDetectorOptions()
        ) -> None:
            """Initialize a TFLite object detection model.
            Args:
                model_path: Path to the TFLite model.
                options: The config to initialize an object detector. (Optional)
            Raises:
                ValueError: If the TFLite model is invalid.
                OSError: If the current OS isn't supported by EdgeTPU.
            """

            # Load metadata from model.
            displayer = metadata.MetadataDisplayer.with_model_file(model_path)

            # Save model metadata for preprocessing later.
            model_metadata = json.loads(displayer.get_metadata_json())
            process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
            mean = 0.0
            std = 1.0
            for option in process_units:
                if option['options_type'] == 'NormalizationOptions':
                    mean = option['options']['mean'][0]
                    std = option['options']['std'][0]
            self._mean = mean
            self._std = std

            # Load label list from metadata.
            file_name = displayer.get_packed_associated_file_list()[0]
            label_map_file = displayer.get_associated_file_buffer(file_name).decode()
            label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
            self._label_list = label_list

            # Initialize TFLite model.
            if options.enable_edgetpu:
                if edgetpu_lib_name() is None:
                    raise OSError("The current OS isn't supported by Coral EdgeTPU.")
                interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[load_delegate(edgetpu_lib_name())],
                    num_threads=options.num_threads)
            else:
                interpreter = Interpreter(
                    model_path=model_path, num_threads=options.num_threads)

            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]

            # From TensorFlow 2.6, the order of the outputs become undefined.
            # Therefore we need to sort the tensor indices of TFLite outputs and to know
            # exactly the meaning of each output tensor. For example, if
            # output indices are [601, 599, 598, 600], tensor names and indices aligned
            # are:
            #   - location: 598
            #   - category: 599
            #   - score: 600
            #   - detection_count: 601
            # because of the op's ports of TFLITE_DETECTION_POST_PROCESS
            # (https://github.com/tensorflow/tensorflow/blob/a4fe268ea084e7d323133ed7b986e0ae259a2bc7/tensorflow/lite/kernels/detection_postprocess.cc#L47-L50).
            sorted_output_indices = sorted(
                [output['index'] for output in interpreter.get_output_details()])
            self._output_indices = {
                self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
                self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
                self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
                self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
            }

            self._input_size = input_detail['shape'][2], input_detail['shape'][1]
            self._is_quantized_input = input_detail['dtype'] == np.uint8
            self._interpreter = interpreter
            self._options = options

        def detect(self, input_image: np.ndarray) -> List[Detection]:
            """Run detection on an input image.
            Args:
                input_image: A [height, width, 3] RGB image. Note that height and width
                can be anything since the image will be immediately resized according
                to the needs of the model within this function.
            Returns:
                A Person instance.
            """
            image_height, image_width, _ = input_image.shape

            input_tensor = self._preprocess(input_image)

            self._set_input_tensor(input_tensor)
            self._interpreter.invoke()

            # Get all output details
            boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
            classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
            scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
            count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

            return self._postprocess(boxes, classes, scores, count, image_width,
                                    image_height)

        def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
            """Preprocess the input image as required by the TFLite model."""

            # Resize the input
            input_tensor = cv2.resize(input_image, self._input_size)

            # Normalize the input if it's a float model (aka. not quantized)
            if not self._is_quantized_input:
                input_tensor = (np.float32(input_tensor) - self._mean) / self._std

            # Add batch dimension
            input_tensor = np.expand_dims(input_tensor, axis=0)

            return input_tensor

        def _set_input_tensor(self, image):
            """Sets the input tensor."""
            tensor_index = self._interpreter.get_input_details()[0]['index']
            input_tensor = self._interpreter.tensor(tensor_index)()[0]
            input_tensor[:, :] = image

        def _get_output_tensor(self, name):
            """Returns the output tensor at the given index."""
            output_index = self._output_indices[name]
            tensor = np.squeeze(self._interpreter.get_tensor(output_index))
            return tensor

        def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                        scores: np.ndarray, count: int, image_width: int,
                        image_height: int) -> List[Detection]:
            """Post-process the output of TFLite model into a list of Detection objects.
            Args:
                boxes: Bounding boxes of detected objects from the TFLite model.
                classes: Class index of the detected objects from the TFLite model.
                scores: Confidence scores of the detected objects from the TFLite model.
                count: Number of detected objects from the TFLite model.
                image_width: Width of the input image.
                image_height: Height of the input image.
            Returns:
                A list of Detection objects detected by the TFLite model.
            """
            results = []

            # Parse the model output into a list of Detection entities.
            for i in range(count):
                if scores[i] >= self._options.score_threshold:
                    y_min, x_min, y_max, x_max = boxes[i]
                    bounding_box = Rect(
                        top=int(y_min * image_height),
                        left=int(x_min * image_width),
                        bottom=int(y_max * image_height),
                        right=int(x_max * image_width))
                    class_id = int(classes[i])
                    category = Category(
                        score=scores[i],
                        label=self._label_list[class_id],  # 0 is reserved for background
                        index=class_id)
                    result = Detection(bounding_box=bounding_box, categories=[category])
                    results.append(result)

            # Sort detection results by score ascending
            sorted_results = sorted(
                results,
                key=lambda detection: detection.categories[0].score,
                reverse=True)

            # Filter out detections in deny list
            filtered_results = sorted_results
            if self._options.label_deny_list is not None:
                filtered_results = list(
                    filter(
                        lambda detection: detection.categories[0].label not in self.
                        _options.label_deny_list, filtered_results))

            # Keep only detections in allow list
            if self._options.label_allow_list is not None:
                filtered_results = list(
                    filter(
                        lambda detection: detection.categories[0].label in self._options.
                        label_allow_list, filtered_results))

            # Only return maximum of max_results detection.
            if self._options.max_results > 0:
                result_count = min(len(filtered_results), self._options.max_results)
                filtered_results = filtered_results[:result_count]

            return filtered_results


    _MARGIN = 10  # pixels
    _ROW_SIZE = 10  # pixels
    _FONT_SIZE = 1
    _FONT_THICKNESS = 1
    _TEXT_COLOR = (0, 0, 255)  # red


    def visualize(
        image: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
            image: The input RGB image.
            detections: The list of all "Detection" entities to be visualize.
        Returns:
            Image with bounding boxes.
        """
        for detection in detections:
            # Draw bounding_box
            start_point = detection.bounding_box.left, detection.bounding_box.top
            end_point = detection.bounding_box.right, detection.bounding_box.bottom
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            class_name = category.label
            probability = round(category.score, 2)
            result_text = class_name + ' (' + str(probability) + ')'
            text_location = (_MARGIN + detection.bounding_box.left,
                            _MARGIN + _ROW_SIZE + detection.bounding_box.top)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

        return image
    
    for item_name in os.listdir(UPLOAD_FOLDER):  # Iterate over all the items in the directory
        item_path = os.path.join(UPLOAD_FOLDER, item_name)  # Get the full path of the item

        if os.path.isfile(item_path):  # Check if the item is a file
            os.remove(item_path)  # Delete the file

        elif os.path.isdir(item_path):  # Check if the item is a folder
            shutil.rmtree(item_path)  # Delete the folder
    
    model_upload=request.files["model_file"]
    filename = model_upload.filename
    model_upload.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    model=os.path.join(os.path.join(UPLOAD_FOLDER, filename))
    
    
    upload_image=request.files["objdet_testimg"]
    
    if upload_image:
        image_name=secure_filename(upload_image.filename)
        upload_image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))

    img= load_img(os.path.join(app.config['UPLOAD_FOLDER'],image_name))


    DETECTION_THRESHOLD = 0.5 #@param {type:"number"}
  

    # !wget -q -O $TEMP_FILE $INPUT_IMAGE_URL
    # image = Image.open(img).convert('RGB')
    image=img.convert('RGB')
    image.thumbnail((512, 512), Image.ANTIALIAS)
    image_np = np.asarray(image)

    # Load the TFLite model
    options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=DETECTION_THRESHOLD,
    )
    detector = ObjectDetector(model_path=model, options=options)

    # Run object detection estimation using the model.
    detections = detector.detect(image_np)

    # Draw keypoints and edges on input image
    image_np = visualize(image_np, detections)

    # Show the detection result
    Image.fromarray(image_np)

    RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    #RGB_img.save(os.path.join(app.config['UPLOAD_FOLDER'], image_name))
    #print(os.getcwd)
    out_path=os.path.join(UPLOAD_FOLDER, image_name) 
    #cv2.imwrite('static/uploads/',RGB_img)
    cv2.imwrite(out_path, RGB_img)
    flash('Test Complete')
    return render_template('objdettest.html', filename=image_name)

@app.route('/display_obj_test/<filename>')
def display_obj_test(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host = '0.0.0.0', port=5000)        

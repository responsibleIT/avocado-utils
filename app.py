from flask import Flask
from flask import request
import os
import shutil
import random
import string
from urllib.request import urlretrieve
from zipfile import ZipFile 

from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
import tensorflow as tf
#specifically, we need tensorflow version 2.13.1 or lower, otherwise tflite will break
assert tf.__version__.startswith('2')
from mediapipe_model_maker import gesture_recognizer

# Configure logging
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

app = Flask(__name__)

# a simple route to check if the server is running
@app.route("/status")
def hello_world():
    return "<h1>Avocado Utils are running!</h1>"

# route for adding metadata from labels.txt to a model for image segmentation
@app.post("/add-metadata/image-segmentation")
def addMetaDataImageSegmentation():
    
    # create a random directory
    random_dir = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    temp_dir = os.path.join('static', random_dir)
    os.mkdir(temp_dir)
    # get location of modelfile and labels.txt from the incoming HTTP request
    label_URL = request.json['labelFile']
    label_file = os.path.join(temp_dir, 'labels.txt')
    model_URL = request.json['modelFile']
    model_file = os.path.join(temp_dir, request.json['modelFileName'])

    # when testing, we are running inside a docker container on windows
    # so we use host.docker.internal to access files from the host's localhost
    # when deployed live, the URL we receive will not contain localhost and nothing happens here
    label_URL = label_URL.replace("localhost", "host.docker.internal")
    model_URL = model_URL.replace("localhost", "host.docker.internal")
    
    app.logger.info('Retrieving label file from: %s', label_URL)
    app.logger.info('Retrieving model file from: %s', model_URL)
    urlretrieve(label_URL, label_file)
    urlretrieve(model_URL, model_file)

    ImageClassifierWriter = image_classifier.MetadataWriter

    # Normalization parameters is required when reprocessing the image. It is
    # optional if the image pixel values are in range of [0, 255] and the input
    # tensor is quantized to uint8. See the introduction for normalization and
    # quantization parameters below for more details.
    # https://www.tensorflow.org/lite/models/convert/metadata#normalization_and_quantization_parameters)
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5

    # Create the metadata writer.
    writer = ImageClassifierWriter.create_for_inference(
        writer_utils.load_file(model_file), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD], [label_file])

    # Populate the metadata into the model.
    export_model_path = os.path.join(temp_dir, request.json['newModelFileName'])
    writer_utils.save_file(writer.populate(), export_model_path)

    # remove the downloaded files
    os.remove(label_file)
    os.remove(model_file)

    HOSTNAME = os.getenv("HOSTNAME")
    export_model_url = HOSTNAME + "/static/" + random_dir + "/"  + request.json['newModelFileName']
    app.logger.info('New model file available at : %s', export_model_url)

    # send back the URL of the converted model file in the HTTP response
    return {
        "url": export_model_url
    }

    # todo: remove the converted model file after a certain period of time

# route for training a model for gesture recognition
@app.post("/train/gesture-recognition")
def trainGestureRecognition():
    
    # create a random directory
    random_dir = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    temp_dir = os.path.join('static', random_dir)
    os.mkdir(temp_dir)
    # get location of zipfile with images from the incoming HTTP request
    zipfile_URL = request.json['zipFile']
    zipfile_file = os.path.join(temp_dir, 'training_data')

    # when testing, we are running inside a docker container on windows
    # so we use host.docker.internal to access files from the host's localhost
    # when deployed live, the URL we receive will not contain localhost and nothing happens here
    zipfile_URL = zipfile_URL.replace("localhost", "host.docker.internal")
        
    app.logger.info('Retrieving zipfile from: %s', zipfile_URL)
    urlretrieve(zipfile_URL, zipfile_file)

    #unzip and delete zip
    with ZipFile(zipfile_file, 'r') as zip_object:
        zip_object.extractall(path=temp_dir) 
    os.remove(zipfile_file)

    # load the dataset, split the dataset: 80% for training, 10% for validation, and 10% for testing
    data = gesture_recognizer.Dataset.from_folder(
        dirname=temp_dir,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    #train the model
    export_path = os.path.join(temp_dir, "exported_model")
    hparams = gesture_recognizer.HParams(export_dir=export_path)
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Export to Tensorflow Lite Model to the export_path
    model.export_model()

    # create another random directory to store the exported model
    random_dir2 = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
    result_dir = os.path.join('static', random_dir2)
    os.mkdir(result_dir)

    # move the trained model to new location for downloading
    os.rename( os.path.join(export_path, "gesture_recognizer.task"),  os.path.join(result_dir, request.json['newModelFileName']) )

    # delete temporary files
    shutil.rmtree(temp_dir)

    # Send a response, we are done
    HOSTNAME = os.getenv("HOSTNAME")
    export_model_url = HOSTNAME + "/static/" + random_dir2 + "/"  + request.json['newModelFileName']
    app.logger.info('New model file available at : %s', export_model_url)
    return {
        "url": export_model_url 
    }

    # TODO remove exported model after a while

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)


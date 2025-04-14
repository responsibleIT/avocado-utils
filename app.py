from flask import Flask
from flask import request
import os
import random
import string
from urllib.request import urlretrieve

from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils

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


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host="0.0.0.0", port=port)


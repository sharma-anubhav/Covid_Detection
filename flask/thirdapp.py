import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def get_model():
    global v19,mnV2,rn50
    v19 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/statistical/v19.h5')
    mnV2 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/statistical/mnv2.h5')
    rn50 = load_model('/Users/anubhavsharma/PycharmProjects/neuralnetwork1/statistical/rn50.h5')
    print(" * Model loaded!")


def preprocess_image(image, target_size):
    IMG_SIZE = 224
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


print(" * Loading Keras model...")
get_model()


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))

    p1 = v19.predict(processed_image).tolist()
    p3 = mnV2.predict(processed_image).tolist()
    p4 = rn50.predict(processed_image).tolist()

    pEnsamble = []
    pEnsamble.append([(p3[0][0]+p4[0][0])/2.0, (p3[0][1]+p4[0][1])/2.0 , (p3[0][2]+p4[0][2])/2.0])
    if pEnsamble[0][1] >= 0.5:
        prediction = pEnsamble
    else:
        prediction = p1
    response = {
        'prediction': {
            'COVID': prediction[0][0],
            'Normal': prediction[0][1],
            'Pneumonia': prediction[0][2]
        }
    }
    return jsonify(response)

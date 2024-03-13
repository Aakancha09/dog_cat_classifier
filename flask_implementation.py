from flask import Flask, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = keras.models.load_model('C:/Users/Aakancha/Downloads/dog_cat_classifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.json:
            return jsonify({"error": "No image path provided in the request body"})

        image_path = request.json['image']

        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img)

        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  

        predictions = model.predict(img_array)

        class_labels = ['cat', 'dog']

        predicted_class_index = np.argmax(predictions, axis=1)

        predicted_class_label = class_labels[predicted_class_index[0]]

        result = {
             "predicted_class": predicted_class_label,
             "class_probabilities": predictions.tolist()
         }

        return jsonify(result)

    except Exception as e:
         return jsonify({"error": str(e)})

if __name__ == '__main__':
     app.run(debug=True)
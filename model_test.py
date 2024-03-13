from PIL import Image
from tensorflow import keras 
from PIL import Image
model = keras.models.load_model('C:/Users/Aakancha/Downloads/dog_cat_classifier.h5')

from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img = image.load_img('C:/Users/Aakancha/Downloads/l.jpeg', target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values

# Make a prediction using the loaded model
predictions = model.predict(img_array)

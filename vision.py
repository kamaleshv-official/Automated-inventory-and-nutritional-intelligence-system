import streamlit as st
from tensorflow.keras.utils import img_to_array
import numpy as np
from keras.models import load_model


@st.cache_resource
def load_vgg_model():
    return load_model("vgg.h5", compile=False)


labels = {
    0: 'Apple', 1: 'Banana', 2: 'Beetroot', 3: 'Bell pepper', 4: 'Cabbage', 5: 'Capsicum',
    6: 'Carrot', 7: 'Cauliflower', 8: 'Chilli pepper', 9: 'Corn', 10: 'Cucumber', 11: 'Eggplant',
    12: 'Garlic', 13: 'Ginger', 14: 'Grapes', 15: 'Jalapeno', 16: 'Kiwi', 17: 'Lemon',
    18: 'Lettuce', 19: 'Mango', 20: 'Onion', 21: 'Orange', 22: 'Paprika', 23: 'Pear',
    24: 'Peas', 25: 'Pineapple', 26: 'Pomegranate', 27: 'Potato', 28: 'Raddish',
    29: 'Soy beans', 30: 'Spinach', 31: 'Sweetcorn', 32: 'Sweetpotato', 33: 'Tomato',
    34: 'Turnip', 35: 'Watermelon'
}

fruits = {
    'Banana', 'Apple', 'Pear', 'Grapes', 'Orange', 'Kiwi', 'Watermelon',
    'Pomegranate', 'Pineapple', 'Mango'
}

vegetables = {
    'Cucumber', 'Carrot', 'Capsicum', 'Onion', 'Potato', 'Lemon', 'Tomato', 'Raddish',
    'Beetroot', 'Cabbage', 'Lettuce', 'Spinach', 'Soy beans', 'Cauliflower', 'Bell pepper',
    'Chilli pepper', 'Turnip', 'Corn', 'Sweetcorn', 'Sweetpotato', 'Paprika',
    'Jalapeno', 'Ginger', 'Garlic', 'Peas', 'Eggplant'
}

def classify_image(img, model):
    """Processes a PIL image and returns the prediction."""
    # Resize the image for VGG-16
    img_resized = img.resize((224, 224))
    
    
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    
    predicted_class = labels[predicted_index]
    confidence = float(prediction[predicted_index])
    
    
    category = "Vegetable" if predicted_class in vegetables else "Fruit"
    
    return predicted_class.capitalize(), category, confidence
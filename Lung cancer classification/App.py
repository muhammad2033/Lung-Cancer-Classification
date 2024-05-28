import streamlit as st
import joblib
from keras.preprocessing.image import load_img, img_to_array

# Load multiple models
model_paths = {
    'LogisticRegression': 'LogisticRegression.pkl'  # Corrected model name
}

loaded_models = {}
for model_name, model_path in model_paths.items():
    try:
        loaded_models[model_name] = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model '{model_name}': {e}")

# Define classes for each model
classes = {
    'RandomForest': ['Lung_opacity', 'Normal', 'Viral Pneumonia'],
    'DecisionTree': ['Lung_opacity', 'Normal', 'Viral Pneumonia'],
    'LogisticRegression': ['Lung_opacity', 'Normal', 'Viral Pneumonia']
}

# Create a sidebar for model selection
selected_model = st.sidebar.selectbox('Select Model', list(model_paths.keys()))

st.title('Image Classification')

# Add a header with model information
st.header(f"Selected Model: {selected_model}")

img = st.file_uploader('Select image', type=['jpg', 'png', 'jpeg'])

if img is not None:
    img = load_img(img, target_size=(100, 100), color_mode='grayscale')
    img_arr = img_to_array(img)
    img_flatten = img_arr.flatten()

    st.write('Flatten shape is ', img_flatten.shape)

    # Use the selected model for prediction
    selected_loaded_model = loaded_models.get(selected_model)
    if selected_loaded_model is not None:
        # Perform prediction
        res = selected_loaded_model.predict(img_flatten.reshape(1, -1))  # Reshape input for prediction  # Reshape input for prediction
        predicted_class = classes[selected_model][res[0]]
        st.write(f'{selected_model} predicted class is: {predicted_class}')
    else:
        st.error("Selected model is not loaded or does not exist.")

    # Display the uploaded image
    st.image(img, caption="Uploaded Image")

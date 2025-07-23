import os
import numpy as np
import sys
import webbrowser
import pickle
from threading import Timer
import warnings
from PIL import Image
from flask import Flask, render_template, request, jsonify
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = 'food_classifier_model_fast.h5'
LABEL_ENCODER_PATH = 'food_label_encoder_fast.pkl'

app = Flask(__name__)
global_model_data = {}

CALORIE_DATABASE = {
    'apple_pie': 237, 'baby_back_ribs': 288, 'baklava': 403, 'beef_carpaccio': 120, 'beef_tartare': 150,
    'beet_salad': 86, 'beignets': 350, 'bibimbap': 560, 'bread_pudding': 160, 'breakfast_burrito': 305,
    'bruschetta': 150, 'caesar_salad': 180, 'cannoli': 340, 'caprese_salad': 140, 'carrot_cake': 414,
    'ceviche': 142, 'cheesecake': 321, 'cheese_plate': 400, 'chicken_curry': 240, 'chicken_quesadilla': 310,
    'chicken_wings': 290, 'chocolate_cake': 370, 'chocolate_mousse': 250, 'churros': 237, 'clam_chowder': 90,
    'club_sandwich': 590, 'crab_cakes': 200, 'creme_brulee': 300, 'croque_madame': 650, 'cup_cakes': 305,
    'deviled_eggs': 60, 'donuts': 452, 'dumplings': 230, 'edamame': 122, 'eggs_benedict': 500,
    'escargots': 90, 'falafel': 333, 'filet_mignon': 270, 'fish_and_chips': 585, 'foie_gras': 462,
    'french_fries': 312, 'french_onion_soup': 70, 'french_toast': 230, 'fried_calamari': 175,
    'fried_rice': 174, 'frozen_yogurt': 159, 'garlic_bread': 350, 'gnocchi': 210, 'greek_salad': 150,
    'grilled_cheese_sandwich': 392, 'grilled_salmon': 208, 'guacamole': 150, 'gyoza': 200, 'hamburger': 295,
    'hot_and_sour_soup': 80, 'hot_dog': 290, 'huevos_rancheros': 400, 'hummus': 166, 'ice_cream': 207,
    'lasagna': 135, 'lobster_bisque': 200, 'lobster_roll_sandwich': 450, 'macaroni_and_cheese': 310,
    'macarons': 384, 'miso_soup': 40, 'mussels': 172, 'nachos': 346, 'omelette': 154, 'onion_rings': 300,
    'oysters': 68, 'pad_thai': 455, 'paella': 340, 'pancakes': 227, 'panna_cotta': 250, 'peking_duck': 337,
    'pho': 400, 'pizza': 266, 'pork_chop': 221, 'poutine': 740, 'prime_rib': 350, 'pulled_pork_sandwich': 420,
    'ramen': 436, 'ravioli': 210, 'red_velvet_cake': 370, 'risotto': 166, 'samosa': 252, 'sashimi': 140,
    'scallops': 111, 'seaweed_salad': 130, 'shrimp_and_grits': 250, 'spaghetti_bolognese': 160,
    'spaghetti_carbonara': 380, 'spring_rolls': 150, 'steak': 271, 'strawberry_shortcake': 320,
    'sushi': 140, 'tacos': 226, 'takoyaki': 200, 'tiramisu': 240, 'tuna_tartare': 180, 'waffles': 291
}


def initialize_and_train_model():
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    DATA_DIRECTORY = os.path.join('food-101', 'images')

    print("--- Model Training Initialized (Fast & Accurate Mode) ---")

    if not os.path.exists(DATA_DIRECTORY):
        sys.exit(f"Error: Directory not found at '{DATA_DIRECTORY}'.")

    filepaths = []
    labels = []
    for food_class in os.listdir(DATA_DIRECTORY):
        class_path = os.path.join(DATA_DIRECTORY, food_class)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                filepaths.append(os.path.join(food_class, image_file))
                labels.append(food_class)

    df = pd.DataFrame({'filepath': filepaths, 'label': labels})

    all_classes = df['label'].unique()
    sampled_classes = np.random.choice(all_classes, size=15, replace=False)

    sampled_df = df[df['label'].isin(sampled_classes)].groupby('label').sample(n=75, random_state=42)

    print(f"Training on {len(sampled_df)} images from {len(sampled_classes)} food categories.")

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=sampled_df,
        directory=DATA_DIRECTORY,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=sampled_df,
        directory=DATA_DIRECTORY,
        x_col='filepath',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    le = LabelEncoder()
    le.classes_ = np.array(list(train_generator.class_indices.keys()))

    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True  # Fine-tune all layers from the start for simplicity and speed

    inputs = Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(le.classes_), activation='softmax')(x)
    model = Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Fine-tuning the model...")
    model.fit(train_generator, validation_data=validation_generator, epochs=10, verbose=1)

    model.save(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)

    print(f"✅ Model trained and saved to '{MODEL_PATH}'")
    return model, le


def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        print(f"Loading pre-trained model from '{MODEL_PATH}'...")
        model = load_model(MODEL_PATH)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            le = pickle.load(f)
    else:
        model, le = initialize_and_train_model()

    global_model_data['model'] = model
    global_model_data['label_encoder'] = le
    print("✅ System is ready.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(img_array_expanded)

        model = global_model_data['model']
        le = global_model_data['label_encoder']

        prediction_vector = model.predict(preprocessed_img, verbose=0)[0]
        predicted_class_index = np.argmax(prediction_vector)
        predicted_label = le.inverse_transform([predicted_class_index])[0]
        confidence = np.max(prediction_vector)

        calories = CALORIE_DATABASE.get(predicted_label, "N/A")

        print(
            f"Prediction: {predicted_label.replace('_', ' ').title()} | Confidence: {confidence:.2%} | Calories: {calories}")

        return jsonify({
            'prediction': predicted_label.replace('_', ' ').title(),
            'confidence': f"{confidence:.2%}",
            'calories': f"{calories} kcal (est. per 100g)" if calories != "N/A" else "Not Available"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == '__main__':
    load_or_train_model()
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False, use_reloader=False)
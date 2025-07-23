# 🍽️ Food Recognition & Calorie Estimation

> 🧠 AI-powered web app that classifies food items and estimates calories from uploaded images — built with deep learning and Flask.

---

## 📌 Overview

**Food Recognition and Calorie Estimation** is a professional-grade AI project designed to identify food from images and estimate calorie content using a trained CNN model on the **Food-101** dataset. Built with Flask and TensorFlow, it represents a complete machine learning pipeline — from model training to responsive web deployment.

---

## ✨ Features

✅ **AI-Based Food Detection**  
Classifies 101 different food items with a fine-tuned CNN.

🔥 **Calorie Estimation Engine**  
Provides estimated calorie content per 100g using an integrated food database.

🖼️ **Image Upload Interface**  
Upload any food photo — get instant predictions with confidence score and calorie estimate.

🌐 **Responsive Web UI**  
Clean, mobile-friendly interface built with Flask + Bootstrap.

🚀 **Optimized Performance**  
Fast, accurate predictions powered by `.h5` model and `.pkl` label encoder.

---

## 🖥️ Demo

Experience the application through these screenshots:

<p align="center">
  <img src="https://github.com/syedmehfooz47/PRODIGY_ML_05/blob/master/demo/127.0.0.1_5000_.png" width="80%" alt="Home Page">
  <br><br>
  <img src="https://github.com/syedmehfooz47/PRODIGY_ML_05/blob/master/demo/127.0.0.1_5000_%20(1).png" width="48%" alt="Upload Page">
  <img src="https://github.com/syedmehfooz47/PRODIGY_ML_05/blob/master/demo/127.0.0.1_5000_%20(2).png" width="48%" alt="Prediction Result">
</p>

---

## 🛠 Tech Stack

| Layer        | Technologies                        |
|--------------|-------------------------------------|
| **Frontend** | HTML, CSS, Bootstrap                |
| **Backend**  | Python, Flask                       |
| **ML Model** | TensorFlow, Keras, Scikit-learn     |
| **Image Utils** | Pillow, NumPy                   |
| **Deployment** | Localhost (extendable to cloud)   |

---

## ⚙️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/syedmehfooz47/PRODIGY_ML_05.git
cd PRODIGY_ML_05
````

2. **Create and Activate Virtual Environment**

For Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the App**

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🚀 Usage Guide

1. Open the app in your browser.
2. Upload an image of food.
3. Get instant predictions:

   * Food Category
   * Confidence %
   * Calories per 100g

---

## 🧠 Model Information

* **Dataset**: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
* **Model**: Convolutional Neural Network
* **Framework**: TensorFlow + Keras
* **Training**: Transfer learning with fine-tuning
* **Outputs**: Class label, confidence, calorie estimate

---

## 📜 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👨‍💻 Author

Developed with passion by
[**Syed Mehfooz C S**](https://github.com/syedmehfooz47)

---

## ⭐ Support the Project

If you found this helpful, please give it a ⭐ on GitHub and share with others!

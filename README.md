
---

## 🧠 Brain Tumor Detection System

### 🌟 Overview

The **Brain Tumor Detection System** is a deep learning-powered web application designed to classify brain MRI images into different tumor types. It uses **transfer learning** with a pre-trained **ResNet50** model to deliver accurate and fast predictions. The app is built with **Streamlit**, offering an intuitive interface for users to upload and analyze MRI scans with ease.

---

### 🚀 Features

* **Brain Tumor Classification**: Detects and classifies brain tumors into:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor
* **Transfer Learning with ResNet50**: Leverages pretrained ImageNet weights for accurate feature extraction.
* **Interactive UI**: Simple and user-friendly web app built using Streamlit.
* **Real-time Inference**: Upload an MRI image and get instant predictions.

---

### 🛠 Installation

**Clone the repository**:

```bash
git clone https://github.com/YourUsername/brain-tumor-detection-app.git
cd brain-tumor-detection-app
```

**Install the required dependencies**:

```bash
pip install -r requirements.txt
```

---

### ▶️ Running the Application

Run the following command in your terminal:

```bash
streamlit run app.py
```

Then, open the URL provided (typically `http://localhost:8501`) in your browser.

---

### 📦 Dependencies

* Python
* TensorFlow
* Streamlit
* NumPy
* PIL (Pillow)
* Matplotlib

---

### 📂 Files

* `app.py` : Main Streamlit application
* `brain_tumor.h5` : Trained ResNet50-based model
* `Brain_tumor_dataset/` : Dataset folder (Training & Testing folders)
* `requirements.txt` : Python dependencies list
* `README.md` : Project documentation

---

### 🎯 Usage

1. Open the Streamlit app in your browser.
2. Upload an MRI scan image in `.jpg`, `.jpeg`, or `.png` format.
3. The app will preprocess the image, run it through the trained model, and display the predicted tumor type.
4. Example classes include **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

---

### 📊 Model Details

* **Base Model**: ResNet50 (ImageNet weights)
* **Input Size**: 224x224
* **Output**: Softmax layer with 4 classes
* **Training**: Performed on a labeled MRI brain tumor dataset
* **Evaluation**: Accuracy and loss visualized over training epochs

---

### 📸 Sample Screenshot

*You can include a screenshot of your UI here*

```bash
# Save a screenshot as demo.png and use the line below:
![App Screenshot](https://github.com/AdityaTagde/Brain_tumor_Detection/blob/main/p1.png)
![App Screenshot](https://github.com/AdityaTagde/Brain_tumor_Detection/blob/main/p2.png)
```

---

### ✅ Future Improvements

* Add Grad-CAM or saliency maps for visual explainability
* Deploy on Hugging Face Spaces or Streamlit Cloud
* Add option for batch predictions

---


### 🙋‍♂️ Author

Developed by [Aditya Tagde]([https://www.linkedin.com/](https://www.linkedin.com/in/aditya-tagde-011989299/))

---



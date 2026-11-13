### 🫁 Pneumonia Prediction from Chest X-Ray Images(Cloud-Based Database)
#### VGG19 | Transfer Learning | Flask | Cloud-Based Database

An end-to-end Deep Learning image classification project that analyzes chest X-ray images and predicts whether an image belongs to the **NORMAL** or **PNEUMONIA** class using a pretrained VGG19 architecture.

The project covers the complete machine learning lifecycle, including dataset analysis, image preprocessing, data augmentation, transfer learning, fine-tuning, model evaluation, model saving, and Flask deployment.

---

### 📌 Project Overview

Pneumonia is a respiratory infection that can cause visible abnormalities in chest X-ray images. This project explores how Deep Learning and Transfer Learning can be used to build a computer-aided image classification system.

The project uses the pretrained **VGG19 (Visual Geometry Group 19)** architecture to classify chest X-ray images into two categories:

- **NORMAL** – No pneumonia detected.
- **PNEUMONIA** – Pneumonia-related patterns detected.

The pretrained VGG19 model is adapted to the chest X-ray classification task using Transfer Learning and Fine-Tuning.

The final trained model is integrated into a Flask web application, allowing users to upload chest X-ray images and receive model-generated predictions.

### End-to-End Workflow

```text
Chest X-Ray Dataset
        |
        v
Exploratory Data Analysis
        |
        v
Image Preprocessing
        |
        v
Data Augmentation
        |
        v
Pretrained VGG19 Architecture
        |
        v
Transfer Learning
        |
        v
Model Training
        |
        v
Fine-Tuning
        |
        v
Model Evaluation
        |
        v
Best Model Selection
        |
        v
Save Trained Model
        |
        v
Flask Web Application
        |
        v
Upload X-Ray Image
        |
        v
Generate Prediction
        |
        v
NORMAL / PNEUMONIA
```

---

## 🎯 Project Objectives

- Build an automated chest X-ray image classification system.
- Perform Exploratory Data Analysis and analyze class distribution.
- Apply image preprocessing and normalization.
- Use data augmentation to improve model generalization.
- Analyze class imbalance and apply class weighting where appropriate.
- Implement the pretrained VGG19 architecture using Transfer Learning.
- Fine-tune selected pretrained VGG19 layers.
- Evaluate the model using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- Analyze model performance and select the best-performing VGG19 model.
- Integrate the trained model into a Flask web application.
- Allow users to upload chest X-ray images through a web interface.
- Generate NORMAL/PNEUMONIA predictions.
- Manage large trained model files using Git LFS.

---

## 🧠 Pretrained Model: VGG19

This project uses **VGG19 (Visual Geometry Group 19)**, a pretrained Convolutional Neural Network developed for image classification.

VGG19 consists of 19 trainable layers, including convolutional and fully connected layers in its original architecture.

The pretrained VGG19 model is used as the foundation for learning image features from chest X-ray images.

### Why VGG19?

- Pretrained on a large-scale image dataset.
- Learns useful visual features through convolutional layers.
- Supports Transfer Learning for medical image classification.
- Provides a feature extraction backbone for custom classification tasks.
- Supports Fine-Tuning to adapt pretrained features to chest X-ray images.

The pretrained VGG19 architecture is adapted to classify images into NORMAL and PNEUMONIA categories.

---

## 🔧 Transfer Learning Strategy

Instead of training a Deep Learning model completely from scratch, the pretrained VGG19 architecture is used as the starting point.

### Stage 1 — Feature Extraction

The pretrained VGG19 convolutional layers are initially frozen while custom classification layers are trained.

```text
Pretrained VGG19
       |
       v
Frozen Convolutional Layers
       |
       v
Feature Extraction
       |
       v
Custom Classification Layers
       |
       v
NORMAL / PNEUMONIA
```

### Stage 2 — Fine-Tuning

Selected pretrained VGG19 layers are unfrozen and trained with a lower learning rate.

```text
Pretrained VGG19
       |
       v
Selected Layers Unfrozen
       |
       v
Fine-Tuning
       |
       v
Custom Classification Head
       |
       v
NORMAL / PNEUMONIA
```

Fine-tuning allows the model to adapt pretrained features to the characteristics of chest X-ray images.

---

## 🔬 Data Preprocessing

Before training, the X-ray images undergo preprocessing to ensure consistency.

### Preprocessing Steps

1. Load chest X-ray images.
2. Resize images to the required input dimensions.
3. Normalize pixel values.
4. Convert images into model-compatible tensors.
5. Create batches for training.
6. Assign corresponding class labels.

### Preprocessing Workflow

```text
Raw X-Ray Image
       |
       v
Image Loading
       |
       v
Image Resizing
       |
       v
Normalization
       |
       v
Tensor Conversion
       |
       v
Model Input
```

---

## 🔄 Data Augmentation

Data augmentation is applied to training images to improve generalization and reduce overfitting.

### Techniques Used / Explored

- Horizontal Flip
- Rotation
- Width Shift
- Height Shift
- Shearing
- Zooming

### Augmentation Workflow

```text
          Original X-Ray
                 |
       +---------+---------+
       |         |         |
       v         v         v
    Rotation    Flip    Shift/Zoom
       |         |         |
       +---------+---------+
                 |
                 v
          Augmented Images
```

Data augmentation increases training-data diversity without requiring additional real-world images.

---

## ⚖️ Class Imbalance Handling

Class imbalance can cause a model to favor the majority class.

The project analyzes the distribution of NORMAL and PNEUMONIA images and can use class weighting during model training.

Example:

```python
class_weights = {
    0: 1.05,
    1: 0.95
}
```

The exact weights depend on the class distribution of the training dataset.

---

## 📊 Model Evaluation

The VGG19 model is evaluated using training and validation metrics to monitor model performance and learning progress.

| Metric | Description |
|---|---|
| Accuracy | Measures the proportion of correctly classified training images. |
| Loss | Measures the model's prediction error during training. |
| Validation Accuracy | Measures classification accuracy on the validation dataset. |
| Learning Rate | Controls the step size used by the optimizer during model training. |

### 📈 Model Performance

The following metrics are monitored during VGG19 model training and validation.

| Model | Accuracy | Loss | Val_Accuracy | Learning Rate |
|---|---:|---:|---:|---:|
| **VGG19** |  0.9511  | — | — | — |


---

## 🌐 Flask Web Application

The trained VGG19 model is integrated into a Flask-based web application.

The application provides a web interface where users can upload chest X-ray images and receive classification results.

### Prediction Workflow

```text
User
  |
  v
Upload Chest X-Ray
  |
  v
Flask Backend
  |
  v
Image Validation
  |
  v
Resize & Normalize
  |
  v
Trained VGG19 Model
  |
  v
Prediction Probability
  |
  v
Classification
  |
  v
NORMAL / PNEUMONIA
```

### Application Steps

1. **Open the application:** Launch the Flask application in your browser.
2. **Upload an X-ray:** Select a chest X-ray image from your computer.
3. **Submit the image:** The image is sent to the Flask backend.
4. **Preprocess the image:** The application applies the preprocessing used during model development.
5. **Generate prediction:** The processed image is passed to the trained VGG19 model.
6. **Display result:** The application displays the predicted class.

### Example Prediction

```text
Prediction: PNEUMONIA
```

or

```text
Prediction: NORMAL
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Pretrained CNN Architecture | VGG19 |
| Image Processing | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Model Evaluation | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Large Model Files | Git LFS |

---

## 📂 Project Structure

```text
Pneumonia_Prediction/
│
├── app.py
│
├── pneumonia_classifier.ipynb
│
├── model.h5
│
├── model_weights/
│   ├── vgg19_model_01.h5
│   ├── vgg19_model_02.h5
│   └── vgg_unfrozen.h5
│
├── static/
│   └── uploads/
│       ├── NORMAL2-IM-0338-0001.jpeg
│       └── person1660_virus_2869.jpeg
│
├── templates/
│   └── ...
│
├── requirements.txt
│
├── .gitignore
│
├── .gitattributes
│
└── README.md
```

Large `.h5` files are managed using Git LFS.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Ruthu543/Pneumonia-Detection-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Pneumonia_Prediction
```

### 3. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
```

Activate the environment:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Git LFS Setup

The trained `.h5` model files can be large, so this project uses Git Large File Storage (Git LFS).

Install and initialize Git LFS:

```bash
git lfs install
```

Pull the tracked model files:

```bash
git lfs pull
```

To verify tracked LFS files:

```bash
git lfs ls-files
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will typically be available at:

```text
http://127.0.0.1:5000
```

Open the address in your browser and upload a chest X-ray image.

---

## 🧪 Training Workflow

The training notebook contains the complete VGG19 model development process.

```text
Import Libraries
       |
       v
Load Dataset
       |
       v
Explore Dataset
       |
       v
Analyze Class Distribution
       |
       v
Visualize Sample Images
       |
       v
Preprocess Images
       |
       v
Apply Data Augmentation
       |
       v
Prepare Data Generators
       |
       v
Load Pretrained VGG19
       |
       v
Build Classification Model
       |
       v
Train Classification Head
       |
       v
Fine-Tune Selected Layers
       |
       v
Evaluate Model
       |
       v
Generate Confusion Matrix
       |
       v
Compare Training Results
       |
       v
Save Best Model
```

---

## 📁 Dataset

The project works with chest X-ray images categorized into two classes:

- NORMAL
- PNEUMONIA

The dataset is divided into training, validation, and testing sets.

### Expected Dataset Structure

```text
dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset itself is not included in this repository.

---

## 🚀 Future Improvements

- Improve model performance through additional fine-tuning.
- Experiment with different VGG19 hyperparameters.
- Improve handling of class imbalance.
- Add Grad-CAM for model explainability.
- Display prediction confidence.
- Add Docker support.
- Deploy the Flask application to a cloud platform.
- Add automated model evaluation.
- Improve the web application's UI/UX.
- Extend the system to support additional chest X-ray conditions.

---

## 💡 Key Learnings

This project provided practical experience in:

- Deep Learning
- CNN-based image classification
- VGG19 Architecture
- Transfer Learning
- Fine-Tuning
- Medical image preprocessing
- Data augmentation
- Class imbalance handling
- TensorFlow and Keras
- Model evaluation
- Flask application development
- Git and GitHub
- Git LFS
- End-to-end ML project development

---

## 🔮 Project Highlights

- End-to-End Deep Learning Pipeline
- Pretrained VGG19 Architecture
- Transfer Learning
- Fine-Tuning
- Data Augmentation
- Class Imbalance Analysis
- Multiple Evaluation Metrics
- Flask Web Application
- Git LFS for Large Models
- Image-Based Prediction

---

## 👨‍💻 Author

**Ruthu Madhavi Kola**

GitHub: [https://github.com/Ruthu543](https://github.com/Ruthu543)

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

---

## 📄 Disclaimer

This project is developed for educational and research purposes. Predictions generated by the model should not be considered medical advice, diagnosis, or treatment recommendations.

Always consult a qualified healthcare professional for medical evaluation.


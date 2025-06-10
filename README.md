# **DeepLeaf: AI-Powered Plant Seedling Classifier Using Optimized CNNs**

**Description**
DeepLeaf is an AI-based system that classifies plant seedlings into 12 different species using a **custom-designed Convolutional Neural Network (CNN)**. Built entirely from scratch without pre-trained models, it is optimized for biological image classification—especially useful in **precision agriculture**, **biodiversity tracking**, and **environmental monitoring**.

---

## 🌿 **Project Overview**

* **Goal**: Accurate classification of 12 plant seedling species from images
* **Architecture**: Custom CNN (no transfer learning)
* **Accuracy**: \~80–85% after 50 epochs
* **Dataset**: Kaggle Plant Seedlings Classification

---

## 🗂️ **Dataset Details**

* **Total Images**: \~4,717 RGB images
* **Image Size**: Resized to 70×70 pixels
* **Classes (12)**:
* Dataset Link: https://www.kaggle.com/c/plant-seedlings-classification/data

  * Black-grass
  * Charlock
  * Cleavers
  * Common Chickweed
  * Common Wheat
  * Fat Hen
  * Loose Silky-bent
  * Maize
  * Scentless Mayweed
  * Shepherd’s Purse
  * Small-flowered Cranesbill
  * Sugar Beet

---

## 🧠 **Model Architecture**

**Custom CNN trained from scratch (not pre-trained)**

* **Convolutional Blocks**:

  * Block 1: Conv2D (64 filters) → BatchNorm → ReLU → MaxPooling → Dropout(0.1)
  * Block 2: Conv2D (128 filters) → BatchNorm → ReLU → MaxPooling → Dropout(0.1)
  * Block 3: Conv2D (256 filters) → BatchNorm → ReLU → MaxPooling → Dropout(0.1)
* **Fully Connected Layers**:

  * Flatten → Dense(512) → Dropout(0.5) → Dense(12, Softmax)
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Epochs**: 50
* **Total Parameters**: \~3.3 million

---

## 🧹 **Preprocessing Steps**

* Image resizing to 70×70
* **Gaussian blur** for noise reduction
* **Color space conversion**: RGB to HSV
* **Green masking** to segment plants
* **Normalization** to \[0, 1]
* **Data Augmentation**: rotation, flips, zoom, and shifts

---

## 📈 **Performance Summary**

* **Final Accuracy**: \~80–85% after 50 epochs
* **Evaluation Metrics**: Accuracy, Confusion Matrix, Visual Predictions
* **Visual Examples**: Correct and misclassified images provided

---

## 🛠️ **Technologies Used**

* Python 3.x
* TensorFlow / Keras
* OpenCV
* NumPy, Matplotlib
* scikit-learn

---

## 🚀 **How to Use**

**1. Clone the repository**

```
git clone https://github.com/yourusername/deepleaf.git  
cd deepleaf
```

**2. Install dependencies**

```
pip install -r requirements.txt
```

**3. Prepare dataset**
Download from Kaggle: **Plant Seedlings Classification**
Place in this folder structure:

```
dataset/
 ├── Black-grass/
 ├── Charlock/
 ├── ...
```

**4. Train the model**

```
python train_model.py
```

**5. Predict on new images**

```
python predict.py
```

---

## 🖼️ **Sample Results**

* Image 1: Actual = Maize, Predicted = Maize, Confidence = 94%
* Image 2: Actual = Charlock, Predicted = Sugar Beet, Confidence = 78%

Confusion matrix and more results are saved in the `/results` folder.

![image](https://github.com/user-attachments/assets/331c8a4e-0480-42e7-b350-285e14587713)



---

## 🔭 **Future Enhancements**

* Early stopping
* Mobile deployment for field usage
* Growth stage detection
* Multispectral image support
* Cloud-based analytics dashboard
* Experimentation with EfficientNet, ResNet (for transfer learning)

---

## 📚 **References**

* Kaggle Plant Seedlings Classification Dataset
* "Deep Learning" by Ian Goodfellow
* "Pattern Recognition and Machine Learning" by Bishop
* Research papers on CNNs in agriculture

---

## 🪪 **License**

This project is licensed under the **MIT License**.



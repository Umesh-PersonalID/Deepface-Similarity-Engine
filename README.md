# Deepface-Similarity-Engine

End-to-end Deep Learning based celebrity face retrieval system using FaceNet/VGGFace, MTCNN, cosine similarity, and Streamlit for real-time celebrity recommendation from uploaded images.

---

# Features

- Face Detection using MTCNN
- Deep Face Embedding Extraction
- Celebrity Similarity Matching
- Cosine Similarity Search
- Real-Time Inference
- Streamlit Web Application
- Transfer Learning with Pretrained Models
- Efficient Embedding Storage using Pickle
- Modular Production-Style Architecture

---

# Tech Stack

## Deep Learning & Computer Vision
- TensorFlow / Keras
- VGGFace / FaceNet
- OpenCV
- MTCNN

## Similarity Search
- Cosine Similarity
- NumPy
- scikit-learn

## Web Application
- Streamlit

## Deployment & Utilities
- Docker
- Pickle
- Pandas


# How It Works

## 1. Face Detection
The uploaded image is processed using MTCNN to detect and crop the face region.

## 2. Embedding Generation
The cropped face is passed through a pretrained VGGFace/FaceNet model to generate a high-dimensional feature embedding.

## 3. Similarity Search
Cosine similarity is computed between the uploaded image embedding and stored celebrity embeddings.

## 4. Recommendation
The celebrity with the highest similarity score is returned to the user.

---

# Model Pipeline

```text
Input Image
     ↓
Face Detection (MTCNN)
     ↓
Face Cropping & Preprocessing
     ↓
Feature Extraction (FaceNet/VGGFace)
     ↓
Embedding Vector Generation
     ↓
Cosine Similarity Search
     ↓
Top Celebrity Match
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Deepface-Similarity-Engine.git
cd Deepface-Similarity-Engine
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux/Mac
```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```
---

# Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```
---

# Dataset

- ~8000 celebrity images
- 100+ Bollywood celebrities
- Images resized to 224x224
- Link (https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset)
- Face embeddings precomputed and stored

---



# License

This project is licensed under the MIT License.

---

# Author

Umesh Ghaskata

Graduate Student | Machine Learning Engineer | Deep Learning Enthusiast

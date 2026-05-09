# Deepface-Similarity-Engine

End-to-end Deep Learning based celebrity face retrieval system using FaceNet/VGGFace, MTCNN, cosine similarity, and Streamlit for real-time celebrity recommendation from uploaded images.

---

# 🚀 Features

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

# 🧠 Tech Stack

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

---

# 📂 Project Architecture

```bash
Deepface-Similarity-Engine/
│
├── app/
│   ├── streamlit_app.py
│   ├── inference.py
│   ├── similarity.py
│   └── utils.py
│
├── models/
│   ├── facenet_model.py
│   ├── vggface_model.py
│   └── mtcnn_detector.py
│
├── embeddings/
│   ├── embeddings.pkl
│   ├── image_paths.pkl
│   └── celebrity_names.pkl
│
├── dataset/
│   ├── raw/
│   ├── processed/
│   └── metadata.csv
│
├── scripts/
│   ├── preprocess_images.py
│   ├── generate_embeddings.py
│   └── bulk_download.py
│
├── notebooks/
│   ├── embedding_visualization.ipynb
│   └── experiments.ipynb
│
├── deployment/
│   ├── Dockerfile
│   └── requirements.txt
│
├── api/
│   └── fastapi_server.py
│
├── README.md
└── LICENSE
```

---

# 🔥 How It Works

## 1. Face Detection
The uploaded image is processed using MTCNN to detect and crop the face region.

## 2. Embedding Generation
The cropped face is passed through a pretrained VGGFace/FaceNet model to generate a high-dimensional feature embedding.

## 3. Similarity Search
Cosine similarity is computed between the uploaded image embedding and stored celebrity embeddings.

## 4. Recommendation
The celebrity with the highest similarity score is returned to the user.

---

# 📊 Model Pipeline

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

# ⚙️ Installation

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

# ▶️ Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

# 🖼️ Dataset

- ~8000 celebrity images
- 100+ Bollywood celebrities
- Images resized to 224x224
- Face embeddings precomputed and stored

---

# 📈 Future Improvements

- FAISS Vector Database Integration
- ArcFace Embedding Model
- Real-Time Webcam Inference
- FastAPI Backend
- Docker Deployment
- AWS/GCP Deployment
- Top-K Similarity Recommendations
- Embedding Visualization using t-SNE
- Multi-face Detection Support

---

# 🧪 Concepts Covered

- Deep Learning
- Transfer Learning
- Face Recognition
- Computer Vision
- Embedding Retrieval
- Vector Similarity Search
- Recommendation Systems
- MLOps Fundamentals
- Streamlit Deployment

---

# 📌 Applications

- Celebrity Look-Alike Systems
- Face Recognition Systems
- AI Recommendation Engines
- Similarity Search Systems
- Multimedia Retrieval Systems

---

# 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository and submit pull requests.

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Umesh Ghaskata

Graduate Student | Machine Learning Engineer | Deep Learning Enthusiast

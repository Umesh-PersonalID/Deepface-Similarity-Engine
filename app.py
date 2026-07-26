import gc
import pickle
import os
from pathlib import Path

import sitecustomize

# ----------------------------
# TensorFlow optimizations
# ----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import streamlit as st
from PIL import Image
import numpy as np
import cv2

from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

# ----------------------------
# Paths
# ----------------------------

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

embeddings_path = Path("embedding.pkl")
filenames_path = Path("filenames.pkl")
ui_shell_path = Path("ui/landing_shell.html")

# ----------------------------
# Streamlit
# ----------------------------

st.set_page_config(
    page_title="Celebrity Face Matcher",
    layout="centered",
)

# ----------------------------
# Cached resources
# ----------------------------

@st.cache_resource
def load_model():
    detector = MTCNN()

    model = VGGFace(
        model="resnet50",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )

    return detector, model


@st.cache_data
def load_artifacts():

    if not embeddings_path.exists() or not filenames_path.exists():
        return None, None

    with open(embeddings_path, "rb") as f:
        feature_list = np.asarray(
            pickle.load(f),
            dtype=np.float32,
        )

    with open(filenames_path, "rb") as f:
        filenames = pickle.load(f)

    return feature_list, filenames


def load_ui_shell():
    if ui_shell_path.exists():
        return ui_shell_path.read_text(encoding="utf-8")
    return ""


# ----------------------------
# Utilities
# ----------------------------

def save_uploaded_image(uploaded_image):

    try:
        path = uploads_dir / Path(uploaded_image.name).name

        with open(path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        return path

    except Exception:
        return None


def extract_features(img_path, model, detector):

    img = cv2.imread(str(img_path))

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detections = detector.detect_faces(img_rgb)

    if len(detections) == 0:
        return None

    x, y, w, h = detections[0]["box"]

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, img_rgb.shape[1])
    y2 = min(y + h, img_rgb.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None

    face = img_rgb[y1:y2, x1:x2]

    face = cv2.resize(face, (224, 224))

    face = face.astype(np.float32)

    face = np.expand_dims(face, axis=0)

    face = preprocess_input(face)

    embedding = model.predict(
        face,
        verbose=0,
    ).flatten()

    del img
    del img_rgb
    del face

    gc.collect()

    return embedding


def recommend(feature_list, features):

    features = features.astype(np.float32)

    features /= np.linalg.norm(features) + 1e-10

    normalized_database = feature_list / (
        np.linalg.norm(feature_list, axis=1, keepdims=True) + 1e-10
    )

    similarity = normalized_database @ features

    return int(np.argmax(similarity))


# ----------------------------
# Load resources
# ----------------------------

detector, model = load_model()

feature_list, filenames = load_artifacts()

if feature_list is None:
    st.error(
        "Missing embedding.pkl or filenames.pkl.\n\nRun data_downloader.py then feature_extractor.py."
    )
    st.stop()

# ----------------------------
# UI
# ----------------------------

st.html(load_ui_shell())

with st.sidebar:
    st.subheader("How it works", anchor=False)
    st.markdown("- Upload a clear face image")
    st.markdown("- Detect the face")
    st.markdown("- Extract facial embedding")
    st.markdown("- Compare against celebrity embeddings")


uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "avif"],
)

if uploaded_image:

    image_path = save_uploaded_image(uploaded_image)

    if image_path is None:
        st.error("Unable to save uploaded image.")
        st.stop()

    display_image = Image.open(uploaded_image)

    with st.spinner("Finding closest celebrity..."):

        features = extract_features(
            image_path,
            model,
            detector,
        )

    if features is None:
        st.error("No face detected.")
        st.stop()

    index = recommend(
        feature_list,
        features,
    )

    actor = " ".join(
        Path(filenames[index]).stem.split("_")
    )

    st.success("Match found!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(display_image)

    with col2:
        st.subheader("Closest Match")
        st.write(f"**{actor}**")
        st.image(filenames[index], width=300)

    del features
    gc.collect()
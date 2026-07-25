from pathlib import Path

import sitecustomize
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

st.set_page_config(
    page_title='Celebrity face matcher',
    layout='centered',
)

detector = MTCNN()
model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
uploads_dir = Path('uploads')
uploads_dir.mkdir(exist_ok=True)
embeddings_path = Path('embedding.pkl')
filenames_path = Path('filenames.pkl')
ui_shell_path = Path('ui/landing_shell.html')


@st.cache_resource
def load_artifacts():
    if not embeddings_path.exists() or not filenames_path.exists():
        return None, None

    with embeddings_path.open('rb') as file_handle:
        feature_list = pickle.load(file_handle)

    with filenames_path.open('rb') as file_handle:
        filenames = pickle.load(file_handle)

    return np.asarray(feature_list), filenames

def load_ui_shell():
    if ui_shell_path.exists():
        return ui_shell_path.read_text(encoding='utf-8')
    return ''

def save_uploaded_image(uploaded_image):
    try:
        with open(uploads_dir / uploaded_image.name,'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)

    if not results:
        return None

    x, y, width, height = results[0]['box']
    x = max(x, 0)
    y = max(y, 0)
    width = max(width, 0)
    height = max(height, 0)

    face = img[y:y + height, x:x + width]

    #  extract its features
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)

    face_array = face_array.astype('float32')

    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list,features):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.html(load_ui_shell())

with st.sidebar:
    st.subheader('How it works', anchor=False)
    st.markdown('- Upload a clear face image')
    st.markdown('- We detect the face and extract features')
    st.markdown('- We compare it with known celebrity embeddings')
    st.caption('Tip: front-facing photos usually give better matches.')

feature_list, filenames = load_artifacts()

if feature_list is None or filenames is None:
    st.error('Missing embedding.pkl or filenames.pkl. Run data_downloader.py, then feature_extractor.py, and restart Streamlit.')
    st.stop()

with st.container(border=True):
    uploaded_image = st.file_uploader(
        'Choose an image',
        type=['jpg', 'jpeg', 'png', 'webp', 'avif'],
        help='Supported formats: JPG, JPEG, PNG, WEBP, AVIF',
    )

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)

        with st.spinner('Analyzing face and finding closest celebrity match...'):
            features = extract_features(str(uploads_dir / uploaded_image.name),model,detector)

        if features is None:
            st.error('No face was detected in the uploaded image.')
            st.stop()

        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(Path(filenames[index_pos]).stem.split('_'))

        st.success('Match generated successfully.')

        col1,col2 = st.columns(2)

        with col1:
            st.subheader('Your uploaded image', anchor=False)
            st.image(display_image)
        with col2:
            st.subheader('Closest match', anchor=False)
            st.write(f'Seems like **{predicted_actor}**')
            st.image(filenames[index_pos],width=300)
    else:
        st.error('Could not save the uploaded file. Please try again.')

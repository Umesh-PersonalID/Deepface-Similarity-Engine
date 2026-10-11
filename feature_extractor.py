from pathlib import Path

import sitecustomize
from deepface import DeepFace
from keras.utils import img_to_array, load_img
import numpy as np
import pickle
from tqdm import tqdm

DATASET_DIR = Path('dataset/Bollywood_celeb_face')
FILENAMES_PATH = Path('filenames.pkl')
EMBEDDINGS_PATH = Path('embedding.pkl')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def feature_extractor(img_path):
    # Instead of default (VGG-Face)
    representation = DeepFace.represent(
        img_path=str(img_path),
        model_name="Facenet",
        enforce_detection=False
    )
    return representation[0]["embedding"]

def build_filenames(dataset_dir):
    filenames = sorted(
        str(path)
        for path in dataset_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not filenames:
        raise FileNotFoundError(
            f'No images were found in {dataset_dir}. Run data_downloader.py first or point DATASET_DIR to your dataset.'
        )

    return filenames


if __name__ == '__main__':
    filenames = build_filenames(DATASET_DIR)

    with FILENAMES_PATH.open('wb') as file_handle:
        pickle.dump(filenames, file_handle)

    features = []

    for file in tqdm(filenames):
        features.append(feature_extractor(file))

    with EMBEDDINGS_PATH.open('wb') as file_handle:
        pickle.dump(features, file_handle)


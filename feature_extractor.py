from pathlib import Path

import sitecustomize
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.utils import img_to_array, load_img
import numpy as np
import pickle
from tqdm import tqdm

DATASET_DIR = Path('dataset/Bollywood_celeb_face')
FILENAMES_PATH = Path('filenames.pkl')
EMBEDDINGS_PATH = Path('embedding.pkl')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')

def feature_extractor(img_path,model):
    img = load_img(img_path,target_size=(224,224))
    img_array = img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

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
        features.append(feature_extractor(file,model))

    with EMBEDDINGS_PATH.open('wb') as file_handle:
        pickle.dump(features, file_handle)


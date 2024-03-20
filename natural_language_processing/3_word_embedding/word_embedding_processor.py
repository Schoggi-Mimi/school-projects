from nltk.tokenize import word_tokenize
import nltk

from gensim.models import Word2Vec, FastText, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


import numpy as np

import torch

from nltk.corpus import stopwords

import requests
import zipfile
import os

def download_glove():
    if all(os.path.exists(os.path.join(os.getcwd(), f'glove.6B.{d}d.txt')) for d in [50, 100, 200, 300]):
        print("GloVe files already exist, no need to download again.")
        return

    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    response = requests.get(url, stream=True)
    save_path = os.path.join(os.getcwd(), 'glove.6B.zip')

    with open(save_path, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)

    # Unzip the file
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())

def convert_glove_to_word2vec(glove_input_file, word2vec_output_file):
    glove2word2vec(glove_input_file, word2vec_output_file)



def generate_y(answer_key, answer_key_mapping):
    if answer_key in answer_key_mapping:
        answer_key = answer_key_mapping[answer_key]
    else:
        raise KeyError(f'Answer key {answer_key} not found in answer key mapping')
    labels = [0, 0, 0, 0]
    labels[answer_key] = 1
    return labels

def process_dataset(dataset, remove_stop_words=True, embedding_model='fasttext'):
    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))

    answer_key_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, '1': 0, '2': 1, '3': 2, '4': 3}

    choices_of_questions = []
    y = []

    for data in dataset:
        try:
            choices_encoded = [word_tokenize(data['question'])] + [word_tokenize(choice) for choice in data['choices']['text']]

            if remove_stop_words:
                choices_encoded = [[word for word in choice if word not in stop_words] for choice in choices_encoded]

            if len(choices_encoded[1:]) != 4:
                print(f'Length of choices is not 4, but {len(choices_encoded[1:])}')
                continue

            mapped_answer_key = generate_y(data['answerKey'], answer_key_mapping)

            choices_of_questions.append(choices_encoded)
            y.append(mapped_answer_key)

        except KeyError as e:
            print(e)
            continue
    
    if len(choices_of_questions) != len(y):
        raise Exception('Length of x and y is not equal')
    
    all_choises_flat = [sentence for sublist in choices_of_questions for sentence in sublist]

    if embedding_model.lower() == 'fasttext':
        model = FastText(all_choises_flat, min_count=1)
        choices_of_questions_embeddings = [[np.mean([model.wv[encoding] for encoding in choice], axis=0) for choice in choices] for choices in choices_of_questions]
    elif embedding_model.lower() == 'word2vec':
        model = Word2Vec(all_choises_flat, min_count=1)
        choices_of_questions_embeddings = [[np.mean([model.wv[encoding] for encoding in choice], axis=0) for choice in choices] for choices in choices_of_questions]
    elif embedding_model.lower() == 'glove':
        download_glove()
        glove_input_file = 'glove.6B.100d.txt'
        word2vec_output_file = 'glove.6B.100d.word2vec.txt'
        convert_glove_to_word2vec(glove_input_file, word2vec_output_file)
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        choices_of_questions_embeddings = [[np.mean([model[encoding.lower()] for encoding in choice if encoding.lower() in model.key_to_index], axis=0).astype(np.float32) if any(encoding.lower() in model.key_to_index for encoding in choice) else np.zeros(model.vector_size).astype(np.float32) for choice in choices] for choices in choices_of_questions]
    else:
        raise ValueError(f'Invalid embedding model: {embedding_model}')

    return torch.tensor(np.array(choices_of_questions_embeddings)), y
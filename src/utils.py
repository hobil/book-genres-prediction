import os
import requests
import io
import zipfile
import pickle
import numpy as np
import pandas as pd
import gensim
import logging
import random

import os
import pickle
import numpy as np


def pickle_partial(obj, filename, parts=1):
    """
    Splits a list or numpy array into given number of parts
    and pickles each of them separately.
    The filename must end with '.pkl'
    """
    if parts == 1:
        pickle.dump(obj, open(filename, 'wb'))
    else:
        filename_base = filename.split('.')[0]
        n = len(obj)
        part_size = int(np.ceil(n / parts))
        logging.debug('part_size: {}'.format(part_size))
        for part in range(parts):
            filename_part = '{}_{}.pkl'.format(filename_base, part)
            pickle.dump(obj[part * part_size : (part + 1) * part_size], open(filename_part, 'wb'))


def unpickle(filename):
    """
    Look for a file with the specified name.
    If it is not found, load files with names in format 'filename_i.pkl'
    starting with i=0 and increment i as long as the corresponding file exists.
    Filename must end with '.pkl'
    """
    if os.path.exists(filename):
        logging.debug('{} found'.format(filename))
        return pickle.load(open(filename,'rb'))
    else:
        filename_base = ".".join(filename.split('.')[:-1]) # part before '.pkl'
        part = 0
        result = None
        obj_type = None
        while True:
            filename_part = '{}_{}.pkl'.format(filename_base, part)
            if not os.path.exists(filename_part):
                logging.debug('{} not found, stopping'.format(filename_part))
                break
            logging.debug('loading {}'.format(filename_part))
            obj = pickle.load(open(filename_part,'rb'))
            if obj_type is None:
                obj_type = type(obj)
                result = obj
            elif isinstance(obj, list):
                result.extend(obj)
            else:
                result = np.append(result, obj)
            part += 1
        return result

def load_snippets(doc_size=200, word_tokenized=False):
    ids = pickle.load(open('../res/documents_ids.pkl','rb'))
    genres_ids = pickle.load(open('../res/genres_snippets.pkl','rb')).index
    genres = pickle.load(open('../res/genres_snippets.pkl','rb'))
    ids_train = pickle.load(open('../res/ids_train_1_genre.pkl','rb'))
    ids_test = pickle.load(open('../res/ids_test_1_genre.pkl','rb'))

    # Prepare labels
    train_genres = genres.loc[ids_train]
    y_train = train_genres.columns[np.argmax(train_genres.values,axis=1)]
    test_genres = genres.loc[ids_test]
    y_test = test_genres.columns[np.argmax(test_genres.values,axis=1)]
    
    if word_tokenized:
        texts = unpickle('../res/documents_{}_word_tokenize_tokens.pkl'.format(snippet_length))
    else:
        texts = unpickle('../res/documents_{}_simple_tokens.pkl'.format(snippet_length))
    texts = np.array(texts)
    
    if snippet_length == 3200:
        ids = np.array(genres_ids)
    else:
        ids = np.array(ids)

    assert ids.shape == texts.shape

    text_id_dict = {i : text for i,text in zip(ids, texts)}

    train_texts = np.array([text_id_dict[i] for i in ids_train])
    print(train_texts.shape)
    test_texts = np.array([text_id_dict[i] for i in ids_test])
    print(test_texts.shape)
    
    return train_texts, test_texts, y_train, y_test


def load_glove_embedding(size=50, output_pickle_file='glove_embeddings_{}.pkl'):
    output_file = output_pickle_file.format(size)
    if os.path.exists(output_file):
        return pickle.load(open(output_file,'rb'))
    else:
        glove_vectors_file = GLOVE_VECTORS_FILE.format(size)
        embeddings_index = {}
        with open(glove_vectors_file, encoding='UTF-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                value = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = value
        pickle.dump(embeddings_index, open(output_file,'wb'))
        return embeddings_index


def load_catalog():
    """
    If the catalog is not stored locally at the given path,
    the catalog is downloaded from the given URL.
    returns: pandas.DataFrame
    """
    METADATA_CATALOG_URL = 'http://www.ms.mff.cuni.cz/~bilekja/metadata_catalog.pkl.zip'
    METADATA_CATALOG_ZIP_PATH = '../res/metadata_catalog.pkl.zip'
    METADATA_CATALOG_PATH = '../res/metadata_catalog.pkl'
    RES = '../res/'

    if not os.path.exists(METADATA_CATALOG_PATH):
        # load the metadata catalog
        print('downloading catalog from {}'.format(METADATA_CATALOG_URL))
        response = requests.get(METADATA_CATALOG_URL)
        z = zipfile.ZipFile(io.BytesIO(response.content))

        # !curl $METADATA_CATALOG_URL -o $METADATA_CATALOG_ZIP_PATH
        # unzip it
        #z2 = zipfile.ZipFile(METADATA_CATALOG_ZIP_PATH)
        z.extractall(RES)

    # unpickle it - the format is dictionary
    metadata_catalog_dict = pickle.load(open(METADATA_CATALOG_PATH,'rb'))
    # convert the catalog to pandas DataFrame for more convenient handling
    df = pd.DataFrame.from_dict(metadata_catalog_dict, orient='index').set_index('id')
    return df


def tokenize_folder(input_folder, output_folder, tokenizer=gensim.utils.simple_preprocess):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_books = len(os.listdir(input_folder))
    i=0
    for filename in os.listdir(input_folder):
        i+=1
        logging.info("\r{} out of {} books processed - {}".format(i, total_books, filename))
        try:
            with open(os.path.join(input_folder,filename), encoding="utf-8") as f:
                doc_id = filename.split('.')[0]
                book_text = f.read()
                # tokenized the book text
                tokens = tokenizer(book_text)
                pickle.dump(tokens, open(os.path.join(output_folder, doc_id), 'wb'))
        except Exception as e:
            logging.warning(e)


def load_corpus(folder, skip_first_n_words = None, words_in_document = None, samples_per_document=1, document_sampling_random=True, random_seed=42):
    random.seed(random_seed)
    n = len(os.listdir(folder))
    for i,doc_id in enumerate(os.listdir(folder)):
        if doc_id[0] == '.':
            continue
        tokens = pickle.load(open(os.path.join(folder, doc_id),'rb'))
        n_tokens = len(tokens)
        print("\r{} / {} documents processed.".format(i,n), end='')
        for sample_id in range(samples_per_document):
            # if words in document is a tuple or list, choose a number from the interval
            if type(words_in_document) in (tuple, list):
                w = random.randint(words_in_document[0], words_in_document[1])
            else:
                w = words_in_document
            # if document_sampling_random, choose start of the document part randomly
            if document_sampling_random:
                if skip_first_n_words > (n_tokens - skip_first_n_words - w):
                    continue
                start_token_id = random.randint(skip_first_n_words, n_tokens - skip_first_n_words - w)
            else:
                start_token_id = skip_first_n_words + sample_id * w
            end_token_id = start_token_id + w
            if end_token_id < len(tokens):
                t = tokens[start_token_id:end_token_id]
                yield gensim.models.doc2vec.TaggedDocument(t, [str(doc_id) + '_' + str(sample_id)])

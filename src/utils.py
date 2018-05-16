import os
import requests
import io
import zipfile
import pickle
import pandas as pd
import gensim
import logging
import random

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

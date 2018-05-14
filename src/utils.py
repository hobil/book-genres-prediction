import os
import requests
import io
import zipfile
import pickle
import pandas as pd

def load_catalog():
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

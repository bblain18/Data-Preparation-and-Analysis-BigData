import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import pandas as pd

_DATA_DESCRIPTION = "NRDC generated dataset for Tensorflow data preparation and analysis. Contains soil temperatures from Western Nevada over the month of September."
_DATA_URL = "sensor.nevada.edu"

_DATA_PATH = "datasets/testDataset.csv"
_DATASET_PATH = "datasets/SeptemberSoilTemperature.gz"

#Data cleaning
def prep_data(_DATA_PATH):
    ######### Data Extraction ###########
    #Extract header block
    headers = pd.read_csv(_DATA_PATH, header=None, nrows=9)
    #Extract dataset
    dataset = pd.read_csv(_DATA_PATH, header=None, skiprows=range(0,9))

    ######### Cleaning Headers ##########
    print(headers)


    ######### Cleaning Dataset ##########
    #Convert data in column 1 to datetime
    dataset[0] = pd.to_datetime(dataset[0])
    #Extract day from datetime in column 1
    dataset[0] = dataset[0].dt.day_process_image_file

    #Convert data in column 2 to float
    dataset[1] = dataset[1].astype(float)

    ########### Output Pandas DF to GZ ############
    dataset.to_csv(_DATASET_PATH, compression='gzip')

#Create a class for the dataset. This class is to be renamed in future iterations
# TODO: Rename class
class testDatasetNRDC(tfds.core.GeneratorBasedBuilder):
    #Start class description

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            #Specifies the object (tfds.core.DatasetInfo)
            builder=self,
            #Description of dataset that will appear on the datasets page
            description=(_DATA_DESCRIPTION),
            features=tfds.features.FeaturesDict({
                "day": tfds.features.ClassLabel(num_classes=31), #Each day can be a label
                "temperature": tfds.features.Text()
            }),
            urls=[_DATA_URL],
            supervised_keys=("day", "temperature")
        )
    def _split_generators(self, dl_manager):
        # Downloads the data and defines splits
        # download and extract URLs
        path = dl_manager.extract({
            'SeptemberSoilTemperature': _DATASET_PATH
        })
        print('run')
        #pass # TODO: define object

    def _generate_examples(self):
        # Yields examples from the Dataset

        yield 'key', {}

import tensorflow_datasets.public_api as tfds
import tensorflow as tf

_DATA_DESCRIPTION = "NRDC generated dataset for Tensorflow data preparation and analysis. Contains soil temperatures from Western Nevada over the month of September."
_DATA_URL = "sensor.nevada.edu"

_DATASET_PATH = "datasets/SeptemberSoilTemperature.gz"

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

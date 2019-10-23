import tensorflow_datasets.public_api as tfds

#Create a class for the dataset. This class is to be renamed in future iterations
# TODO: Rename class
class MyDataset(tfds.core.GeneratorBasedBuilder):
    #Start class description

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            #Specifies the object (tfds.core.DatasetInfo)
            builder=self,
            #Description of dataset that will appear on the datasets page
            description=("NRDC generated dataset for Tensorflow data preparation and analysis"),
            urls=['sensor.nevada.edu']
        )

    def _split_generators(self, dl_manager):
        # Downloads the data and defines splits
        # download and extract URLs
        dl_paths = dl_manager.manual_dir #Use manually downloaded data due to option selection on site
        #pass # TODO: define object

    def _generate_examples(self):
        # Yields examples from the Dataset
        yield 'key', {}

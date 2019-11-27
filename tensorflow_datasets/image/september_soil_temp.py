import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import tensorflow.contrib.factorization as tffact
import os as os
import pandas as pd


_DATA_DESCRIPTION = "NRDC generated dataset for Tensorflow data preparation and analysis. Contains soil temperatures from Western Nevada over the month of September."
_DATA_URL = "sensor.nevada.edu"

_DATASET_PATH = "SeptemberSoilTemperature.csv"

#Create a class for the dataset. This class is to be renamed in future iterations
# TODO: Rename class
class septemberSoilTemp(tfds.core.GeneratorBasedBuilder):
    #Start class description

    VERSION = tfds.core.Version('0.1.1')

    def _info(self):
        return tfds.core.DatasetInfo(
            #Specifies the object (tfds.core.DatasetInfo)
            builder=self,
            #Description of dataset that will appear on the datasets page
            description=(_DATA_DESCRIPTION),
            features=tfds.features.FeaturesDict({
                "day": tfds.features.ClassLabel(names=["1","2","3","4","5","6","7","8","9","10","11","12","13","14",
                                                    "15","16","17","18","19","20","21","22","23"]), #Each day can be a label
                "temperature": tfds.features.Tensor(shape=(), dtype=tf.float32)
            }),
            urls=[_DATA_URL],
            supervised_keys=("day", "temperature")
        )
    def _split_generators(self, dl_manager):
        # Downloads the data and defines splits
        # download and extract URLs
        path = os.path.join(dl_manager.manual_dir,
                                  _DATASET_PATH)
        df = pd.read_csv(path)

        if not tf.io.gfile.exists(path):
            # The current celebahq generation code depends on a concrete version of
            # pillow library and cannot be easily ported into tfds.
            msg = "You must download the dataset files manually and place them in: "
            msg += dl_manager.manual_dir
            msg += " as .gz files. Current path read: "
            raise AssertionError(msg)

        day = df.pop('day')
        temp = df.pop('temp')

        dataset = tf.data.Dataset.from_tensor_slices((day.values, temp.values))

        return[
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=1,
                gen_kwargs={
                    "archive": dataset}
                )
        ]
        #pass # TODO: define object

    def _generate_examples(self, archive):
        # # Yields examples from the Dataset
        # # with tf.io.gfile.GFil(archive, 'rb') as file:
        # #     dataset =
        for feat, targ in archive:
            record = {
                "day": feat,
                "temperature": targ,
                }
        yield feat, record

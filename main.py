import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
#import tensorflow_datasets.image.septermberSoiltemp.py

_DATA_PATH = "datasets/testDataset.csv"
_DATASET_PATH = "SeptemberSoilTemperature.csv"

#Data cleaning
def prep_data():
    ######### Data Extraction ###########
    #Extract header block
    headers = pd.read_csv(_DATA_PATH, header=None, nrows=9)
    #Extract dataset
    dataset = pd.read_csv(_DATA_PATH, header=None, skiprows=range(0,9), names=["day", "temp"])

    ######### Cleaning Headers ##########
    print(headers)


    ######### Cleaning Dataset ##########
    #Convert data in column 1 to datetime
    dataset["day"] = pd.to_datetime(dataset["day"])
    #Extract day from datetime in column 1
    dataset["day"] = pd.DatetimeIndex(dataset["day"]).day

    #Convert data in column 2 to float
    dataset["temp"] = dataset["temp"].astype(float)

    ########### Output Pandas DF to GZ ############
    dataset.to_csv(_DATASET_PATH, index=False)
    return dataset["temp"].tolist()

def update(aggregate, newData):
    (count, mean, M2) = aggregate
    count += 1
    delta = newData - mean
    mean += delta / count
    delta2 = newData - mean
    M2 += delta * delta2

    return (count, mean, M2)

def finalize(aggregate):
    (count, mean, M2) = aggregate
    (mean, variance) = (mean, M2 / count)
    if count < 2:
        return float('nan')
    else:
        return (mean, variance)

def findAvgVar(data):
    aggregate = (0, 0, 0)
    analysis = (0, 0)
    for temp in data:
        aggregate = update(aggregate, temp)
    analysis = finalize(aggregate)
    print (analysis)

def find_std():
    print("std")

def find_mad():
    print("std")

def find_aad():
    print("std")

def find_distribution():
    print("std")

def main():
    tf.enable_eager_execution()
    #Load data
    sst_train, info = tfds.load("september_soil_temp", split="train", with_info=True)
    #Shuffle batch
    sst_train_batch = sst_train.shuffle(20).padded_batch(240, padded_shapes = {"day": [], "temperature": []})
    # Assert instance
    assert isinstance(sst_train_batch, tf.data.Dataset)
    # Numpy matrix vars
    label = []
    image = []
    # Take matrix from batch
    for sst_example in sst_train_batch:
        image, label = sst_example["day"], sst_example["temperature"]

    # Analysis
    mean = np.mean(label.numpy())
    variance = np.var(label.numpy())
    std = np.std(label.numpy())

    print("Mean: {}".format(mean))
    print("Variance: {}".format(variance))
    print("Standard Deviation: {}".format(std))


main()

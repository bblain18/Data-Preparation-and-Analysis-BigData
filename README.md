NRDC Data Preparation and Analysis Using Tensorflow:

Project designed using Python scripts and Tensorflow. Dataset consists of Soil Temperatures in Eastern Nevada for the month of September.

Next step: Make dataextraction.py interpret. Test data can be found under tensorflow_datasets/manual

Project Description: Make a sensor dataset for Tensorflow and perform simple analysis. Extract data from http://sensor.nevada.edu/SENSORDataSearch/, where several different sensor data is available and you can choose one or more datasets. Then use Tensorflow’s API to create a dataset that is ready for Tensorflow: https://www.tensorflow.org/datasets/add_dataset. Please make sure you clean the data first. Please also make some simple analysis for the data as well: e.g., compute the mean, standard deviation, ADD, MAD, and distribution. Please also use visualization to demonstrate the data and your analysis results. Note: wind speed prediction is NOT allowed as it has been used in Data Mining class.

NRDC Data Preparation and Analysis Using Tensorflow

Project designed using Python scripts and Tensorflow.

Notes: Data set: Soil temperature of some location for the month of September (9) Labels: Day Features/Values: Temperature

Data preparation:

[x] Create and label values

[x] Iterate over rows in CSV & split "labels" and "values".
Split rows based upon ',' into an array This array will have a label at [0] and value at [1] with our particular dataset

[x] Form train and test splits


Data cleaning:

[x] Remove all null values/replace them with 0s

[x] Remove Time, Data, Year from data/row

[x] (should be able to use .Month)

[x] Ensure temperature/row[1] is a float

[x] Remove file "headers" (ex: Timestamp, Measurement interval, etc) from dataset and output to log for analysis

Analysis:

- Avg/day

[x] Avg over month

- Variance/day

- Variance over month

- Std Deviation/day

- Std Deviation over month

- MADD/day

- MADD over month

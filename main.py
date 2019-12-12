import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#import tensorflow_datasets.image.septermberSoiltemp.py

_DATA_PATH = "datasets/testDataset2.csv"
_DATASET_PATH = "SeptemberSoilTemperature.csv"
headers = []

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

def findTempSetForDay(targetDay, batch_day, batch_temp):
    index = np.where(batch_day.numpy() == targetDay)
    return batch_temp.numpy()[index]

################# DAILY ANALYSIS ########################################
def plotDailyAnalysis(days, avgs, min, max, var, trend, vtd, fullMonthInfo, all_val, all_days):
    fig = plt.figure()
    analysisPlot = fig.add_subplot(221)
    analysisTrend = fig.add_subplot(222)
    analysisVariance = fig.add_subplot(223)
    # analysis.scatter(all_days, all_val)
    analysisPlot.plot(days, avgs, label='Mean temperature')
    analysisPlot.plot(days, min, label='Minimum temperature')
    analysisPlot.plot(days, max, label='Maximum temperature')
    analysisPlot.scatter(days, avgs, label='Mean temperature')
    analysisPlot.scatter(days, min, label='Minimum temperature')
    analysisPlot.scatter(days, max, label='Maximum temperature')
    analysisPlot.set_xlabel('Day')
    analysisPlot.set_ylabel('Temperature degC')
    analysisPlot.set_title('Soil Temperatures per Day')

    analysisTrend.bar(days, trend, label='Change by day')
    analysisTrend.plot(days, vtd, label='Variance to date', color='orange')
    analysisTrend.set_xlabel('Day')
    analysisTrend.set_ylabel('Percent change')
    analysisTrend.set_title('Temperature Trend')

    analysisVariance.bar(days, var, label='Variance by day')
    analysisVariance.set_xlabel('Day')
    analysisVariance.set_ylabel('Temperature degC')
    analysisVariance.set_title('Temperature Variance over Day')


    fig.tight_layout()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.7, 0.3, fullMonthInfo, fontsize=14,
        verticalalignment='top', bbox=props)
    # plt.subplots_adjust(left=0.3)
    plt.suptitle('September Soil Temperature Trends')
    plt.show()

def dailyAnalysis(batch_day, batch_temp):
    list_days = []
    list_avg = []
    list_min = []
    list_max = []
    originalDay = 0
    currentavg = 0
    list_vtd = []
    list_trend = []
    list_variance = []

    for day in range(1, 31):
        dailySet = findTempSetForDay(day, batch_day, batch_temp)
        if dailySet.size > 0:
            list_avg.append(np.mean(dailySet))
            list_min.append(np.min(dailySet))
            list_max.append(np.max(dailySet))
            list_variance.append(np.var(dailySet))
        else:
            list_avg.append(0)
            list_min.append(0)
            list_max.append(0)
            list_variance.append(0)
        if day is 1:
            list_trend.append(originalDay)
            list_vtd.append(originalDay)
        else:
            list_trend.append((list_avg[-1]-list_avg[-2])/list_avg[-2])
            currentavg = np.mean(list_avg)
            list_vtd.append((list_avg[-1]-currentavg)/currentavg)
        list_days.append(day)

    fullMonthInfo = fullMonthAnalysis(batch_temp)
    plotDailyAnalysis(list_days, list_avg, list_min, list_max, list_variance, list_trend, list_vtd, fullMonthInfo, batch_temp, batch_day)
##################################################################

def fullMonthAnalysis(batch_temp):
    # Analysis
    mean = np.mean(batch_temp.numpy())
    variance = np.var(batch_temp.numpy())
    std = np.std(batch_temp.numpy())
    mad = sm.robust.scale.mad(batch_temp.numpy())

    return '\n'.join((
        r'Mean=%.4f' % (mean, ),
        r'Variance=%.4f' % (variance, ),
        r'Standard Deviation=%.4f' % (std, ),
        r'Mad=%.4f' % (mad, )))

def main():
    tf.enable_eager_execution()
    #Load data
    sst_train, info = tfds.load("september_soil_temp", split="train", with_info=True)
    #Shuffle batch
    sst_train_batch = sst_train.shuffle(20).padded_batch(743, padded_shapes = {"day": [], "temperature": []})
    # Assert instance
    assert isinstance(sst_train_batch, tf.data.Dataset)
    # Numpy matrix vars
    temperature = []
    day = []
    # Take matrix from batch
    for sst_example in sst_train_batch:
        day, temperature = sst_example["day"], sst_example["temperature"]

    dailyAnalysis(day, temperature)

# prep_data()
main()

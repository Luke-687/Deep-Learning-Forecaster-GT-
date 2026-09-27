#Collect the indiviudal numpy arrays from 2000 to 2020 and organize into  single array to be read in neural network training
#Data from 2021 serves as the set of testing cases, model trained on the 20 years prior to predict the snow of 2021
import numpy as np
import os
import argparse

def main():
    inputs = buildingParser().parse_args()

    #Range of years to pull training data from
    yearStart = inputs.train_year_start
    yearEnd = inputs.train_year_end
    #Rage of years to pull testing data from
    yearStartTest = inputs.test_year_start
    yearEndTest = inputs.test_year_end

    #Loop through the year level numpy arrays of training and testing weather data to form completed numpy arrays
    allTrainingWeatherData = np.array([], dtype=object)
    for i in range(yearEnd-yearStart):
        year = yearStart+i
        tempWeatherArray = np.load(f"../3-Define-Data-Pipeline/Yearly-Weather-Storage/mtWashingtonAllWeather{year}.npy", allow_pickle=True)
        allTrainingWeatherData = np.concatenate((allTrainingWeatherData, tempWeatherArray))
    allTestingWeatherData = np.array([], dtype=object)
    for j in range(yearEndTest-yearStartTest):
        year = yearStartTest+j
        tempWeatherArray = np.load(f"../3-Define-Data-Pipeline/Yearly-Weather-Storage/mtWashingtonAllWeather{year}.npy", allow_pickle=True)
        allTestingWeatherData = np.concatenate((allTestingWeatherData, tempWeatherArray))

    return allTrainingWeatherData, allTestingWeatherData, yearStart, yearEnd, yearStartTest, yearEndTest

def buildingParser():
    parser = argparse.ArgumentParser(description='Takes input of year data will be accessed from for CDS and GHCND API calls')
    parser.add_argument('--train_year_start', type=int, default=2000)
    parser.add_argument('--train_year_end', type=int, default=2021)
    parser.add_argument('--test_year_start', type=int, default=2021)
    parser.add_argument('--test_year_end', type=int, default=2022)
    return parser

#All weather from 20 years of API calls organized now into a single array, store for training
if __name__ == '__main__':
    trainingData, testingData, trainYearS, trainYearE, testYearS, testYearE = main()
    AllTrainingTestingData = 'Training-Testing-Data'
    os.makedirs(AllTrainingTestingData, exist_ok=True)
    np.save(os.path.join(AllTrainingTestingData, f'trainingWeather({trainYearS}|{trainYearE}).npy'), trainingData, allow_pickle=True)
    np.save(os.path.join(AllTrainingTestingData, f'testingWeather({testYearS}|{testYearE}).npy'), testingData, allow_pickle=True)

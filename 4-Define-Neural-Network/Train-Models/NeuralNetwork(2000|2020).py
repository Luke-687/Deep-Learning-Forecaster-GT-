#Creator: Luke O'Callaghan
#Notes: Initial snow forecasting network, trained on 2000-2020 daily samples and tested against the 2021 snow record
#Architecture: [56 Weather Metrics] -> Hidden layer 1 -> Hidden layer 2 -> [Snow Output, Confidence]

#Necessary libraries
import numpy as np
import torch
import torch.nn as nn
import os
import argparse

#Resolve data off this file so the script runs from any working directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#The data and model folders are siblings of this script's folder, not children of it
NETWORK_ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(NETWORK_ROOT, 'Training-Testing-Data')
MODEL_DIR = os.path.join(NETWORK_ROOT, 'Save-Trained-Models')

#Column layout of one daily sample as OrganizeAllData.py stores it
#[0] date | [1:57] 56 CDS metrics (4 timestamps x 14 variables) | [57] TMIN | [58] TMAX | [59] SNOW
DATE_COL = 0
CDS_START, CDS_END = 1, 57
SNOW_COL = 59
INPUT_METRICS = 56

#14 variables recorded at each of the 4 daily timestamps, in the order YearSnowMetricPull.py appends them
VARIABLE_ORDER = ['T_850', 'T_700', 'T_500', 'RH_700', 'Q_700', 'Z_500', 'U_850', 'V_850',
                  'T_2m', 'Td_2m', 'MSLP', 'U_10', 'V_10', 'TCWV']

#Name every input column so a trained model can be inspected later
def featureNames():
    return [f'{var}_t{t}' for t in range(4) for var in VARIABLE_ORDER]

# **** DATA LOADING ****

#Pull one stored array apart into the metric matrix, the snow record, and the year of each sample
def unpackSamples(rows):
    weatherMetrics = np.array([row[CDS_START:CDS_END] for row in rows], dtype=np.float32)
    snowRecord = np.array([row[SNOW_COL] for row in rows], dtype=np.float32)
    sampleYears = np.array([row[DATE_COL].year for row in rows], dtype=np.int32)
    return weatherMetrics, snowRecord, sampleYears

#Load the combined arrays and split the training years into a fitting set and a validation set
def loadData(validationStartYear):
    trainingPath = os.path.join(DATA_DIR, 'trainingWeather(2000|2021).npy')
    testingPath = os.path.join(DATA_DIR, 'testingWeather(2021|2022).npy')
    for path in (trainingPath, testingPath):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing combined array, run OrganizeAllData.py first: {path}')

    #allow_pickle is required, each row is a python list holding a date object
    trainingRows = np.load(trainingPath, allow_pickle=True)
    testingRows = np.load(testingPath, allow_pickle=True)

    allMetrics, allSnow, allYears = unpackSamples(trainingRows)
    testMetrics, testSnow, _ = unpackSamples(testingRows)

    #Hold out whole years rather than random days, consecutive days are far too correlated to split randomly
    fitMask = allYears < validationStartYear
    validationMask = allYears >= validationStartYear
    if not fitMask.any() or not validationMask.any():
        raise ValueError(f'validationStartYear={validationStartYear} leaves one of the splits empty')

    return (allMetrics[fitMask], allSnow[fitMask],
            allMetrics[validationMask], allSnow[validationMask],
            testMetrics, testSnow)

# **** FEATURE AND TARGET PREPARATION ****

#Metric scales span eight orders of magnitude (Q_700 ~0.002 against MSLP ~101600), standardize or the large columns drown the rest
def fitNormalizer(fitMetrics):
    metricMeans = fitMetrics.mean(axis=0)
    metricDeviations = fitMetrics.std(axis=0)
    #A constant column would divide by zero
    metricDeviations[metricDeviations < 1e-8] = 1.0
    return metricMeans, metricDeviations

def applyNormalizer(weatherMetrics, metricMeans, metricDeviations):
    return (weatherMetrics - metricMeans) / metricDeviations

#Two targets per day, did it snow at all and how much fell, snowfall is log compressed because the tail reaches 617mm
def buildTargets(snowRecord):
    didSnow = (snowRecord > 0).astype(np.float32)
    snowAmount = np.log1p(snowRecord).astype(np.float32)
    return didSnow, snowAmount

# **** NETWORK DEFINITION ****

class SnowForecastNetwork(nn.Module):
    #Shared trunk learns the weather pattern once, then two heads read a snow chance and a snow total off it
    def __init__(self, inputMetrics=INPUT_METRICS, hiddenOne=128, hiddenTwo=64, dropoutRate=0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(inputMetrics, hiddenOne),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
            nn.Linear(hiddenOne, hiddenTwo),
            nn.ReLU(),
            nn.Dropout(dropoutRate),
        )
        #Raw logit for snow occurrence, sigmoid is applied at prediction time by the loss or by predictSnow
        self.occurrenceHead = nn.Linear(hiddenTwo, 1)
        #Predicts log1p(mm) on days that actually snow
        self.amountHead = nn.Linear(hiddenTwo, 1)

    def forward(self, weatherMetrics):
        sharedFeatures = self.trunk(weatherMetrics)
        occurrenceLogit = self.occurrenceHead(sharedFeatures).squeeze(-1)
        amountPrediction = self.amountHead(sharedFeatures).squeeze(-1)
        return occurrenceLogit, amountPrediction

#Turn the two raw heads into a snow total in mm and a confidence reading between 0 and 1
def predictSnow(network, normalizedMetrics):
    network.eval()
    with torch.no_grad():
        occurrenceLogit, amountPrediction = network(normalizedMetrics)
        snowChance = torch.sigmoid(occurrenceLogit)
        #Undo the log1p compression, clamp because a slightly negative prediction is not physical
        amountIfSnowing = torch.expm1(amountPrediction).clamp(min=0.0)
        #Expected snowfall weights the predicted total by how likely any snow is at all
        expectedSnow = snowChance * amountIfSnowing
    return expectedSnow.numpy(), snowChance.numpy(), amountIfSnowing.numpy()

# **** LOSS ****

#Occurrence is scored on every day, amount only on days that actually snowed, a zero day has no amount to learn
def combinedLoss(occurrenceLogit, amountPrediction, didSnow, snowAmount, occurrenceLossFn, amountLossFn):
    occurrenceLoss = occurrenceLossFn(occurrenceLogit, didSnow)
    snowingDays = didSnow > 0
    if snowingDays.any():
        amountLoss = amountLossFn(amountPrediction[snowingDays], snowAmount[snowingDays])
    else:
        amountLoss = torch.zeros((), device=occurrenceLoss.device)
    return occurrenceLoss + amountLoss, occurrenceLoss, amountLoss

# **** TRAINING ****

def trainNetwork(network, fitTensors, validationTensors, inputs):
    fitMetrics, fitDidSnow, fitAmount = fitTensors
    validationMetrics, validationDidSnow, validationAmount = validationTensors

    optimizer = torch.optim.Adam(network.parameters(), lr=inputs.learning_rate, weight_decay=inputs.weight_decay)
    occurrenceLossFn = nn.BCEWithLogitsLoss()
    amountLossFn = nn.MSELoss()

    trainingSet = torch.utils.data.TensorDataset(fitMetrics, fitDidSnow, fitAmount)
    trainingBatches = torch.utils.data.DataLoader(trainingSet, batch_size=inputs.batch_size, shuffle=True)

    #Early stopping keeps the weights from the best validation epoch rather than the last one
    bestValidationLoss = float('inf')
    bestNetworkState = None
    bestEpoch = -1
    epochsWithoutImprovement = 0

    for epoch in range(inputs.epochs):
        network.train()
        for batchMetrics, batchDidSnow, batchAmount in trainingBatches:
            optimizer.zero_grad()
            occurrenceLogit, amountPrediction = network(batchMetrics)
            batchLoss, _, _ = combinedLoss(occurrenceLogit, amountPrediction, batchDidSnow, batchAmount,
                                           occurrenceLossFn, amountLossFn)
            batchLoss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            occurrenceLogit, amountPrediction = network(validationMetrics)
            validationLoss, occurrenceLoss, amountLoss = combinedLoss(
                occurrenceLogit, amountPrediction, validationDidSnow, validationAmount,
                occurrenceLossFn, amountLossFn)
        validationLoss = validationLoss.item()

        if(validationLoss < bestValidationLoss - 1e-4):
            bestValidationLoss = validationLoss
            bestNetworkState = {key: value.clone() for key, value in network.state_dict().items()}
            bestEpoch = epoch
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1

        if(epoch % 10 == 0 or epochsWithoutImprovement >= inputs.patience):
            print(f'  epoch {epoch:3d}  validation loss {validationLoss:.4f} '
                  f'(occurrence {occurrenceLoss.item():.4f}, amount {amountLoss.item():.4f})')

        if(epochsWithoutImprovement >= inputs.patience):
            print(f'  early stop at epoch {epoch}, no improvement for {inputs.patience} epochs')
            break

    #Restore the best weights before the network is scored or saved
    if bestNetworkState is not None:
        network.load_state_dict(bestNetworkState)
    print(f'  best epoch {bestEpoch}, validation loss {bestValidationLoss:.4f}')
    return network, bestValidationLoss, bestEpoch

# **** EVALUATION ****

#Trivial predictors the network has to beat before it can be called useful
def baselineScores(fitSnow, testSnow):
    alwaysZero = np.zeros_like(testSnow)
    trainingMean = np.full_like(testSnow, fitSnow.mean())
    return {
        'always zero': (np.abs(alwaysZero - testSnow).mean(), np.sqrt(((alwaysZero - testSnow) ** 2).mean())),
        'training mean': (np.abs(trainingMean - testSnow).mean(), np.sqrt(((trainingMean - testSnow) ** 2).mean())),
    }

def evaluateNetwork(network, testMetrics, testSnow, fitSnow):
    expectedSnow, snowChance, _ = predictSnow(network, testMetrics)
    didSnow = (testSnow > 0).astype(np.int32)
    predictedSnow = (snowChance >= 0.5).astype(np.int32)

    #Snow amount accuracy in real millimetres
    meanAbsoluteError = np.abs(expectedSnow - testSnow).mean()
    rootMeanSquaredError = np.sqrt(((expectedSnow - testSnow) ** 2).mean())

    #Snow occurrence accuracy, plus a brier score reading how honest the confidence output is
    truePositives = int(((predictedSnow == 1) & (didSnow == 1)).sum())
    falsePositives = int(((predictedSnow == 1) & (didSnow == 0)).sum())
    falseNegatives = int(((predictedSnow == 0) & (didSnow == 1)).sum())
    accuracy = float((predictedSnow == didSnow).mean())
    precision = truePositives / max(truePositives + falsePositives, 1)
    recall = truePositives / max(truePositives + falseNegatives, 1)
    f1Score = (2 * precision * recall / max(precision + recall, 1e-9))
    brierScore = float(((snowChance - didSnow) ** 2).mean())

    print('\n**** TEST RESULTS ****')
    print(f'Snow amount   MAE  {meanAbsoluteError:7.2f} mm')
    print(f'Snow amount   RMSE {rootMeanSquaredError:7.2f} mm')
    for name, (baselineMae, baselineRmse) in baselineScores(fitSnow, testSnow).items():
        print(f'  vs {name:14s} MAE {baselineMae:7.2f} mm   RMSE {baselineRmse:7.2f} mm')
    print(f'Snow occurrence  accuracy {accuracy:.3f}  precision {precision:.3f}  recall {recall:.3f}  f1 {f1Score:.3f}')
    print(f'Confidence calibration  brier score {brierScore:.4f} (lower is better, 0.25 is a coin flip)')

    return {'mae': float(meanAbsoluteError), 'rmse': float(rootMeanSquaredError),
            'accuracy': accuracy, 'precision': precision, 'recall': recall,
            'f1': f1Score, 'brier': brierScore}

# **** SAVING AND LOADING ****

#The normalizer belongs in the checkpoint, a model saved without it cannot score new weather correctly
def saveNetwork(network, metricMeans, metricDeviations, inputs, testScores, bestEpoch):
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpointPath = os.path.join(MODEL_DIR, f'snowNetwork({inputs.train_label}).pt')
    torch.save({
        'networkState': network.state_dict(),
        'metricMeans': metricMeans,
        'metricDeviations': metricDeviations,
        'featureNames': featureNames(),
        'architecture': {'inputMetrics': INPUT_METRICS, 'hiddenOne': inputs.hidden_one,
                         'hiddenTwo': inputs.hidden_two, 'dropoutRate': inputs.dropout},
        'bestEpoch': bestEpoch,
        'testScores': testScores,
    }, checkpointPath)
    print(f'\nSaved trained network to {checkpointPath}')
    return checkpointPath

#Rebuild a ready to use network and its normalizer from a checkpoint
def loadNetwork(checkpointPath):
    checkpoint = torch.load(checkpointPath, weights_only=False)
    architecture = checkpoint['architecture']
    network = SnowForecastNetwork(architecture['inputMetrics'], architecture['hiddenOne'],
                                  architecture['hiddenTwo'], architecture['dropoutRate'])
    network.load_state_dict(checkpoint['networkState'])
    network.eval()
    return network, checkpoint['metricMeans'], checkpoint['metricDeviations']

# **** MAIN ****

def main():
    inputs = buildingParser().parse_args()
    torch.manual_seed(inputs.seed)
    np.random.seed(inputs.seed)

    print(f'Loading data, validating on {inputs.validation_start_year} onward')
    fitMetrics, fitSnow, validationMetrics, validationSnow, testMetrics, testSnow = loadData(inputs.validation_start_year)
    print(f'  fitting   {len(fitSnow):5d} days')
    print(f'  validation{len(validationSnow):5d} days')
    print(f'  testing   {len(testSnow):5d} days')

    #Fit the normalizer on the fitting split only, validation and test must stay unseen
    metricMeans, metricDeviations = fitNormalizer(fitMetrics)

    def toTensors(weatherMetrics, snowRecord):
        didSnow, snowAmount = buildTargets(snowRecord)
        return (torch.tensor(applyNormalizer(weatherMetrics, metricMeans, metricDeviations)),
                torch.tensor(didSnow), torch.tensor(snowAmount))

    fitTensors = toTensors(fitMetrics, fitSnow)
    validationTensors = toTensors(validationMetrics, validationSnow)
    testTensors = toTensors(testMetrics, testSnow)

    print('\nTraining network')
    network = SnowForecastNetwork(INPUT_METRICS, inputs.hidden_one, inputs.hidden_two, inputs.dropout)
    network, bestValidationLoss, bestEpoch = trainNetwork(network, fitTensors, validationTensors, inputs)

    testScores = evaluateNetwork(network, testTensors[0], testSnow, fitSnow)

    return network, metricMeans, metricDeviations, inputs, testScores, bestEpoch

def buildingParser():
    parser = argparse.ArgumentParser(description='Trains the initial snow forecasting network on combined yearly weather arrays')
    parser.add_argument('--validation_start_year', type=int, default=2017, help='training years from here on are held out for validation')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=25, help='epochs without validation improvement before stopping')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_one', type=int, default=128)
    parser.add_argument('--hidden_two', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_label', type=str, default='2000|2020')
    return parser

if __name__ == '__main__':
    trainedNetwork, means, deviations, parsedInputs, scores, bestEpochIndex = main()
    saveNetwork(trainedNetwork, means, deviations, parsedInputs, scores, bestEpochIndex)

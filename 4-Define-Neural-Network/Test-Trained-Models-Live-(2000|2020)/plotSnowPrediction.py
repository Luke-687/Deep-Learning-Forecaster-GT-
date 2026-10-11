#Creator: Luke O'Callaghan
#Notes: Plots the rolling snow predictions as a bar chart and saves it outside the git directory
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import date

#Anchor every path to this file, a relative savefig would otherwise land wherever the script was launched from
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_ROOT = os.path.dirname(THIS_DIR)
PROJECT_DIR = os.path.dirname(NETWORK_ROOT)
#Generated-Figures sits one level above the git directory so plotted output is never committed
FIGURE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), 'Generated-Figures', 'Plotted-Snow-Prediction')
PREDICTION_FILE = os.path.join(THIS_DIR, 'rollingSnowPredictions(2000|2020).npy')

#Each prediction holds 4 values, [date, expected snow, amount if snowing, chance of snow]
VALUES_PER_PREDICTION = 4

#The rolling file is 2D when freshly written but np.append in the live script flattens it to 1D,
#so accept either shape and always hand back one row per day
def loadPredictions(predictionPath):
    if not os.path.isfile(predictionPath):
        raise FileNotFoundError(f'No rolling predictions yet, run LiveTestingFromWeatherModel first: {predictionPath}')
    #allow_pickle is required, every row carries a date object
    storedPredictions = np.load(predictionPath, allow_pickle=True)
    if storedPredictions.ndim == 1:
        if len(storedPredictions) % VALUES_PER_PREDICTION != 0:
            raise ValueError(f'Flat prediction file holds {len(storedPredictions)} values, not a multiple of {VALUES_PER_PREDICTION}')
        storedPredictions = storedPredictions.reshape(-1, VALUES_PER_PREDICTION)
    return storedPredictions

#Keep only today and the days ahead of it, and sort so the bars read left to right in time
def upcomingPredictions(storedPredictions):
    today = date.today()
    upcoming = [row for row in storedPredictions if row[0] >= today]
    upcoming.sort(key=lambda row: row[0])
    return upcoming

def main():
    storedPredictions = loadPredictions(PREDICTION_FILE)
    upcoming = upcomingPredictions(storedPredictions)
    if len(upcoming) == 0:
        raise RuntimeError('Every stored prediction is in the past, rerun LiveTestingFromWeatherModel to refresh the window')

    #Keep the float values, int() would round 5.08 mm down to 5 mm and lose the small totals entirely
    forecastDates = [row[0] for row in upcoming]
    expectedSnow = [float(row[1]) for row in upcoming]
    snowChance = [float(row[3]) for row in upcoming]
    dayLabels = [forecastDate.strftime('%m/%d') for forecastDate in forecastDates]

    figure, axes = plt.subplots(figsize=(max(6, 0.9 * len(upcoming)), 5))
    bars = axes.bar(dayLabels, expectedSnow, color='tab:blue')

    #Label each bar with its millimetre total and how confident the network is in it
    for bar, snowTotal, chance in zip(bars, expectedSnow, snowChance):
        axes.annotate(f'{snowTotal:.1f} mm\n{chance:.0%}',
                      xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 4), textcoords='offset points',
                      ha='center', va='bottom', fontsize=8)

    axes.set_title(f'Snow Projection, Mt Washington Summit\n{forecastDates[0]} to {forecastDates[-1]}')
    axes.set_xlabel('Forecast day')
    axes.set_ylabel('Expected snowfall (mm)')
    #Headroom so the annotations above the tallest bar are not clipped
    axes.set_ylim(0, max(max(expectedSnow) * 1.25, 1.0))
    axes.grid(axis='y', alpha=0.3)
    axes.set_axisbelow(True)

    #Create the output folder if this is the first plot, savefig will not make it
    os.makedirs(FIGURE_DIR, exist_ok=True)
    #Name the file from the dates actually plotted rather than a hardcoded window
    figureName = f'nextDays({forecastDates[0]:%m_%d_%y}|{forecastDates[-1]:%m_%d_%y}).png'
    figurePath = os.path.join(FIGURE_DIR, figureName)
    figure.savefig(figurePath, dpi=300, bbox_inches='tight')
    plt.close(figure)

    print(f'Plotted {len(upcoming)} day(s), {forecastDates[0]} to {forecastDates[-1]}')
    for forecastDate, snowTotal, chance in zip(forecastDates, expectedSnow, snowChance):
        print(f'  {forecastDate}  {snowTotal:6.2f} mm  {chance:5.1%} chance')
    print(f'Saved figure to {figurePath}')
    return figurePath

if __name__ == '__main__':
    main()

Architecture:
[60 Weather Metrics] -> Hidden layer 1 -> Hidden layer 2 -> [Snow Output, Confidence]

Data Collection Framework:
Collect data hourly from ERA5:
-Upper atmospheric column data
-Geopotential information for precipitation indicators
-2m temperatures and dew point 

Collect generalized daily information from NOAA:
Total daily snowfall

Strecth the data collection timeline as far back as possible, pulling information from countless years of weather samples and generalizations to establish a large training base of the nueral network.

Model will process live weather data and output snow predictions by taking in a 60 metric long input vector and outputting a snow prediction with a confidence reading. Live column and surface data projections will be sourced from Open-Meteo at set intervals, establishing a rolling timeline of data points and results which can recycle into the system for additional learning.

Open-Meteo provides column and surface projections on a future timeline projecting over 2 weeks into the future, making it possible to predict long term snow trends including weekly snow total predictions. These predictions could prove essential for long term planning in harsh snow climates.


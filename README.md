# calibrebio-analyzer

This is some Python code to help analyse the data from the Calibrio Bio Met Cart. Provided in the hope it proves
useful to someone.

© David George, 2025

## Setup

Create a python virtual environment

$ python3 -m venv venv

Install the dependencies

$ pip install -r requirements.txt

## Usage

Show the Heart Rate, Respiratory Volume, O2 consumption the CO2 exhalation

$ python cpet.py --input data/Nov-01-2025-1104-cycling-ramp-test.1.csv  --show HR VE VO2 VCO2  --mark-vt

![VO2 VCO2 HR VE](example-graphs/vo2vco2.png?raw=true "VO2 VCO2 HR VE")

Parameters:

--show
* VE - Respiratory volume
* VO2 - Oxygen consumption
* VCO2 - Carbon dioxide exhalation
* HR - Heart Rate
* POWER - Power data, if present, see sync.py
* VEQO2 - equivalent O2 - VE/VO2
* VEQCO2 - equivalent V)@ - VE/VCO2

--mark-vt calculate ventilatory thresholds 

--smooth-sec smoothing in seconds

--time start,end - trim start and end of graph by X seconds, useful to remove warmup or cooldown data

### Fatmax

Fat burning or fat oxidation (the term preferred by scientists) is the maximum amount of fat that an
athlete can “burn” per hour. It is often expressed in energy (kcal) per hour.

At low intensities you will not burn a lot of fat, since you don’t burn that much energy at all.
* When intensity increases, fat oxidation increases as well. But only up to a point (= FatMax).
* At very high intensities, you will not burn any fat.
* As a result, you get a concave (upward; n-shaped) parabola:

python   cpet.py --input ramp-test.csv --fatmax  --smooth-bpm 10 --trim 30, --no-points

![Fatmax](example-graphs/fatmax.png?raw=true "Fatmax")

--no-points - don't show individual points
--smooth-bmp - smooth hr data
--fatmax - show fatmax plot

### VO2 Max

python cpet.py --input ramp-test.csv --vo2kg --x time  --mark-peaks --title "Mass-specific O2"

![VO2 max](example-graphs/vo2.png?raw=true "VO2 max")

--x
* time
* hr - heart rate

--both show both VO2 Max and VCO2 Max

--mark-peaks - show VO2 max on graph

## VSLOPE

Produce a vslope graph of VO2/VCO2, can help identify ventilatory thresholds. Slope of 1:1 is LV1

python   cpet.py --input ramp-test.csv --vslope  --smooth-sec 10 --trim 30, --color HR

![VSLOPE](example-graphs/vslope.png?raw=true "Vslope")

## Sync

sync.py is a basis utility which takes a tcx file from a bike computer with power data and syncs with a Calibre
file using heart rate to add power data in watts. It is work in progress as the output doesn't include all the
Calibre data it is necessary to cut and paste the power data into a Calibre file with Excel.

python sync.py --cpet example.csv --tcx ride.tcx --out merged.csv

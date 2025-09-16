# CarbonX
Repository for CarbonX

## Supported Models and Model Requirements
1. Sundial:
<br> To run Sundial, first create a virtual environment and install the required modules:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install transformers==4.40.1
pip3 install pandas numpy json5 torch
```

## Supported Features and Grids
Currently, this version only supports point-value forecasting for 2 grids (California and Texas). 

The supported date range is from 2021-01-01 to 2024-12-31.

We are incorporating all the features and will release it soon.

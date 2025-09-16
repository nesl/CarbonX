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
<br> Then, run the CarbonX file with the desired API (you need to manually uncomment the API calls from within the file).
```
python3 carbonx.py 
```

## Supported Features and Grids
While we have locally implemented all the papers mentioned in the paper, most of them need cleaning up before they can be released for public use. At present, the following are supported:

### Supported features:
1. Point-value forecasting: :white_check_mark:
2. Extended forecasting horizon: :white_check_mark:

### Supported grids:
1. California (US-CAL-CISO): :white_check_mark:
2. Texas (US-TEX-ERCO): :white_check_mark:

The supported date range is from 2021-01-01 to 2024-12-31.

We are working on incorporating all the features mentioned in the paper and will release it soon.

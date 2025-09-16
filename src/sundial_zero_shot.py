
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import csv

CONTINENTS = ["Africa", "Asia", "Europe", "North_America", "Central_America", "South_America", "Oceania"]
# CONTINENTS = ["North_America"]
# CONTINENTS = ["Europe", "Central_America"]
# CONTINENTS = ["South_America", "Oceania"]
CONTINENTS = ["Asia", "Africa"]


TIER1_TEST_DAYS = 439
FORECAST_LEN = 96

def getTier1Forecasts(tsfmModel, data):
    print(data.columns)
    ci = data["Unscaled GT"].values
    lastDay = ci[-FORECAST_LEN:]
    ci = ci.reshape(-1, 96)[:, :24].flatten()
    ci = np.concatenate([ci, lastDay[24:]])
    dates = data["Datetime (UTC)"].values
    lastDay = dates[-FORECAST_LEN:]
    dates = dates.reshape(-1, 96)[:, :24].flatten()
    dates = np.concatenate([dates, lastDay[24:]])
    test_ci = ci[-TIER1_TEST_DAYS*24:]
    test_dates = dates[-TIER1_TEST_DAYS*24:]
    output = []
    ciOutput = []
    for i in range(0, len(test_dates) - FORECAST_LEN + 1, 24):
        window = test_dates[i:i + FORECAST_LEN]
        output.extend(window)  # flatten as you go
        ciWindow = test_ci[i:i + FORECAST_LEN]
        ciOutput.extend(ciWindow)
    dates_reshaped = np.array(output)
    ci_reshaped = np.array(ciOutput)
    mape = []
    tier1Forecast = []
    N = len(ci)
    for i in range(N-TIER1_TEST_DAYS*24, N-3*24, 24):
    
        lookback_length = 30*24 
        lookback = torch.tensor(ci[i-lookback_length:i]).unsqueeze(0).float()
        # forecasting configurations
        num_samples = 20           # generate 20 samples
        forecast = tsfmModel.generate(lookback, max_new_tokens=FORECAST_LEN, num_samples=num_samples) # generate 20 probable predictions
        # use raw predictions for mean/quantiles/confidence-interval estimation

        meanForecast = forecast[0].mean(dim=0)
        meanForecast = meanForecast.cpu().numpy()
        tier1Forecast.extend(meanForecast)
        mape.append(round(np.mean(np.abs(ci[i:i+FORECAST_LEN]-meanForecast)*100/ci[i:i+FORECAST_LEN]), 3))
        print(i//24, dates[i].split(" ")[0], round(np.mean(np.abs(ci[i:i+FORECAST_LEN]-meanForecast)*100/ci[i:i+FORECAST_LEN]), 3))

    tier1Forecast = np.array(tier1Forecast)
    mape = np.array(mape)
    print("Tier 1 MAPE: ", np.mean(mape))
    print("90th MAPE: ", np.percentile(mape, 90))

    combined = np.column_stack((dates_reshaped, ci_reshaped, tier1Forecast))
    return combined, np.mean(mape), np.percentile(mape, 90)


if __name__ == "__main__":
    tsfmModel = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True) 
    INPUT_DIR = "./fine_tune_continent_codec/moment/"
    count = 0
    for continent in CONTINENTS:
        allMape = {}
        print(continent)
        for filename in os.listdir(f"{INPUT_DIR}/{continent}/"):
            if ("csv" not in filename):
                continue
            region = filename.split("_")[0]
            print(filename, region)
            data = pd.read_csv(f"{INPUT_DIR}/{continent}/{filename}")
            result, mape, mape90 = getTier1Forecasts(tsfmModel, data)
            np.savetxt(f"./sundial-global-results/{continent}/{region}-tier1.csv", result, delimiter=",", header="Datetime (UTC),True,Tier1-pred", comments='', fmt='%s')
            count +=1
            allMape[region] = (mape, mape90)
        with open(f"./sundial-global-results/{continent}/{continent}_summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["region", "mape", "mape90"])
            for k, (v1, v2) in allMape.items():
                writer.writerow([k, v1, v2])
    print(count)

        # data = pd.read_csv(f"sota-data/{region}/{region}_lifecycle_emissions.csv")
        

    

        
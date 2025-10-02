import numpy as np
import pandas as pd
import json5, json
import os
from datetime import datetime, timedelta
import models
import torch

CONFIG_FILE_DIR = "../config/"
DATA_FILE_DIR = "../data/forecasting-data/"
FORECAST_FILE_DIR = "../ci-forecasts/"
SUCCESS = 0
FAILURE = 1

class CarbonX():
    def __init__(self):
        self.config_file = f"./{CONFIG_FILE_DIR}/config.json"
        self.config = self._get_config()
        self.model_dict = self._initialize_models()

    def get_ci_historical(self, region, date):
        if (region not in self.config["SUPPORTED_REGIONS_FORECASTING"]):
            print("Region not supported!")
            exit(0)
        
        if (self._check_date_validity(date) == FAILURE):
            print("Out of supported date range!")
            exit(0)

        data = pd.read_csv(f"{DATA_FILE_DIR}/{region}.csv", index_col=["Datetime (UTC)"])
        ci_column = data.filter(like="Carbon").columns[0]
        ci_data = data[[ci_column]]
        required_ci_data = ci_data[ci_data.index.str.contains(date, case=False)]
        # print(required_ci_data)
        return required_ci_data

    def get_ci_forecasts(self, region, date, horizon=96, pi=False):
        if (region not in self.config["SUPPORTED_REGIONS_FORECASTING"]):
            print("Region not supported!")
            exit(0)
        
        if (self._check_forecast_date_validity(date) == FAILURE):
            print("Out of supported forecast date range!")
            exit(0)

        data = pd.read_csv(f"{DATA_FILE_DIR}/{region}.csv", index_col=["Datetime (UTC)"])
        ci_column = data.filter(like="Carbon").columns[0]
        gt_ci_data = data.loc[:date][[ci_column]]
        # print(gt_ci_data, "available GT")
        
        cur_model = self._get_model()
        model_reg_entry = self.config["MODEL_REGISTRY"].get(cur_model)
        model_name = model_reg_entry.get("name")

        if (model_name == "Sundial"): # check if forecasts are already available in file
            forecast_file = f"{FORECAST_FILE_DIR}/{model_name}/{region}.csv"
            if (os.path.exists(forecast_file)): 
                forecast_data = pd.read_csv(forecast_file, index_col = ["Datetime (UTC)"])
                forecast_start_date = self.config["FORECAST_DATE_RANGE_START"]
                num_days = self._get_num_days_between_two_dates(forecast_start_date, date)
                ci_column = forecast_data.filter(like="Carbon").columns[0]
                forecast_ci_data = forecast_data.iloc[num_days*horizon:][[ci_column]]
                print("File data found")
                return forecast_ci_data[ci_column].values[:horizon]
            else:
                print("File not found. Generating forecasts on the fly...")
        lookback = model_reg_entry.get("lookback")
        num_samples = model_reg_entry.get("num_samples")
        ci_forecasts = self._get_forecasts(self.model_dict[cur_model], gt_ci_data, 
                                                       lookback=lookback, num_samples=num_samples,
                                                       horizon=horizon)
        
        return ci_forecasts
    
    def get_missing_ci_data(self):
        return
    
    def get_supported_grids(self):
        return self.config["SUPPORTED_REGIONS_FORECASTING"]
    
    def get_forecasting_accuracy(self, region, date):
        print(region, date)
        if (region not in self.config["SUPPORTED_REGIONS_FORECASTING"]):
            print("Region not supported!")
            exit(0)
        
        if (self._check_date_validity(date) == FAILURE):
            print("Out of supported date range!")
            exit(0)

        d = datetime.strptime(date, "%Y-%m-%d")
        days = [d + timedelta(days=i) for i in range(0, 4)]
        days_str = [day.strftime("%Y-%m-%d") for day in days]
        actual = []
        for d in days_str:
            ci_data = self.get_ci_historical(region=region, date=d)
            ci_column = ci_data.filter(like="Carbon").columns[0]
            ci_data = ci_data[ci_column].values
            actual.extend(ci_data)
        forecast = self.get_ci_forecasts(region=region, date=date, horizon=96)
        return self._get_mape(actual, forecast)
    
    def _check_date_validity(self, date):
        date_range_start = datetime.strptime(self.config["DATE_RANGE_START"], "%Y-%m-%d")
        date_range_end = datetime.strptime(self.config["DATE_RANGE_END"], "%Y-%m-%d")
        cur_date = datetime.strptime(date, "%Y-%m-%d")
        if (cur_date >= date_range_start and cur_date <= date_range_end):
            return SUCCESS
        return FAILURE

    def _check_forecast_date_validity(self, date):
        date_range_start = datetime.strptime(self.config["FORECAST_DATE_RANGE_START"], "%Y-%m-%d")
        date_range_end = datetime.strptime(self.config["FORECAST_DATE_RANGE_END"], "%Y-%m-%d")
        cur_date = datetime.strptime(date, "%Y-%m-%d")
        if (cur_date >= date_range_start and cur_date <= date_range_end):
            return SUCCESS
        return FAILURE
    
    def _get_num_days_between_two_dates(self, date1, date2):
        d1 = pd.to_datetime(date1)
        d2 = pd.to_datetime(date2)
        num_days = (d2.date() - d1.date()).days
        return num_days
    
    def _get_forecasts(self, model, ci_data, lookback, num_samples=None, horizon=96):
        ci_column = ci_data.filter(like="Carbon intensity").columns[0]
        ci_data = ci_data[ci_column].values
        N = len(ci_data)
        historical_ci = torch.tensor(ci_data[N-lookback:N]).unsqueeze(0).float()
        forecast = model._get_forecasts(historical_ci, horizon, num_samples)
        return forecast
    
    # Mean Absolute Percentage Error (in %) with zero-safe denominator.
    def _get_mape(self, actual, forecast, eps=1e-8):
        actual = np.asarray(actual, dtype=np.float32)
        forecast = np.asarray(forecast, dtype=np.float32)
        denom = np.maximum(np.abs(actual), eps)
        return float(np.mean(np.abs((actual - forecast) / denom)) * 100.0)
    
    def _get_model(self):
        # print("Current Model: ", self.config["CURRENT_MODEL"])
        return self.config["CURRENT_MODEL"]
    
    def _get_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json5.load(f)
        else:
            print("Config file does not exist")
            exit(0)
        return config
    
    def _initialize_models(self):
        model_dict = {}
        supported_models = self.config["SUPPORTED_MODELS"]
        model_registry = self.config["MODEL_REGISTRY"]
        current_model = self.config["CURRENT_MODEL"]
        for model in supported_models:
            if (model != current_model):
                continue
            reg_entry = model_registry.get(model)
            if not reg_entry:
                print(f"[WARN] No registry entry for '{model}'. Skipping.")
                continue
            model_name = reg_entry.get("name")
            repo = reg_entry.get("repo")
            if not repo:
                print(f"[WARN] Registry entry for '{model}' missing 'repo'. Skipping.")
                continue
            device = reg_entry.get("device")
            if (not torch.cuda.is_available()):
                device = "cpu"
            torch.device(device)
            model_kwargs = dict(reg_entry.get("model_kwargs", {}))
            other_kwargs = {
                "lookback": reg_entry.get("lookback"),
                "num_samples": reg_entry.get("num_samples"), 
                "device": reg_entry.get("device"),
                "forecast_horizon": reg_entry.get("forecast_horizon")
            }
            try:
                print(f"Loading model {model_name}...")
                model_class = getattr(models, model_name)
                model_dict[model] = model_class(repo, model_kwargs, other_kwargs)
            except Exception as e:
                print(f"[ERROR] Failed to load '{model}' ({repo}): {e}")
                exit(0)
        return model_dict
        
    
    # Updates config file with new config.
    def _set_config(self, config):
        with open(f"{CONFIG_FILE_DIR}/{self.config_file}", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return SUCCESS

    # Sets a particular model either in zero-shot or fine-tuned mode.
    def set_model(self, model_name="sundial", mode="zs"):
        new_model = f"{model_name}_{mode}"
        if (new_model in self.config["SUPPORTED_MODELS"]):
            self.config["CURRENT_MODEL"] = new_model
            if (self._set_config(self.config) == FAILURE):
                print("Updating config failed!")
                return FAILURE
        else:
            print("Model/mode not supported!")
            return FAILURE
        self.model_dict = self._initialize_models()
        return SUCCESS
    
if __name__ == "__main__":
    cx = CarbonX()
    # cx._get_model()
    # hist_ci = cx.get_ci_historical("US-CAL-CISO", "2023-01-01")
    # print(hist_ci)
    ci_forecast = cx.get_ci_forecasts("US-CAL-CISO", "2023-01-01")
    print(ci_forecast)
    # print(cx.get_forecasting_accuracy("US-TEX-ERCO", "2021-10-31"))

    # cur_model = cx._get_model()
    # cur_model_name = cx.config["MODEL_REGISTRY"][cur_model]["name"]
    # print(cur_model_name)
    # start_s = "2021-02-01"
    # end_s   = "2024-12-28"
    # days = pd.date_range(start_s, end_s, freq="D")
    # day_to_96hrs = {
    #     d.strftime("%Y-%m-%d"): pd.date_range(d, periods=96, freq="h").strftime("%Y-%m-%d %H:%M").tolist()
    #     for d in days
    # }

    # supported_grids = cx.get_supported_grids()
    # for grid in supported_grids:
    #     forecast_dates = []
    #     ci_forecasts = []
    #     print(grid)
    #     for cur_day in day_to_96hrs.keys():
    #         # print(cur_day)
    #         ci_forecast = cx.get_ci_forecasts(grid, cur_day)
    #         forecast_dates.extend(day_to_96hrs[cur_day])
    #         ci_forecasts.extend(ci_forecast)
        
    #     forecast_df = pd.DataFrame({"Datetime (UTC)": forecast_dates, 
    #                                 "Carbon Intensity Forecast": ci_forecasts})
    #     forecast_df.to_csv(f"{FORECAST_FILE_DIR}/{cur_model_name}/{grid}.csv", index=False)
    #     print(len(forecast_df))



    

  
    

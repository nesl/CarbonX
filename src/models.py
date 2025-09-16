from transformers import AutoModelForCausalLM

class Sundial:
    def __init__(self, repo, model_kwargs):
        self.repo = repo
        self.model_kwargs = model_kwargs
        self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)

    def _get_forecasts(self, input_data, horizon=96, num_samples=20):
        forecast = self.model.generate(input_data, max_new_tokens=horizon, num_samples=num_samples)
        meanForecast = forecast[0].mean(dim=0)
        meanForecast = meanForecast.cpu().numpy()
        return meanForecast
    
class Moment:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class TimeMoe:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class Chronos:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass

class TimesFM:
    def __init__(self, repo, model_kwargs):
        pass

    def _get_forecasts(self, input_data, horizon=96, num_samples=None):
        pass
from gluonts.dataset import common
from gluonts.model import deepar
from gluonts.trainer import Trainer
from gluonts.evaluation import Evaluator
import confidence
import numpy as np
import crps

import pandas as pd
import matplotlib.pyplot as plt

"Load electricity dataset"

Edf = pd.read_csv('datasets/electricity/electricity.csv', header=0, index_col=0)
# url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
# df = pd.read_csv(url, header=0, index_col=0)
data = common.ListDataset([{
    "start": Edf.index[0],
    "target": Edf.MT_003[:44100]
}],
                          freq="5min")

trainer = Trainer(epochs=10)
estimator = deepar.DeepAREstimator(
    freq="5min", prediction_length=24, trainer=trainer)
predictor = estimator.train(training_data=data)

actual = Edf.MT_003[44100:44124].values.tolist()
prediction = next(predictor.predict(data))
eval = Evaluator()
forecast = prediction.mean

print(prediction.mean)
print("MAPE: ", eval.mape(actual, forecast))
print("MSE: ", eval.mse(actual, forecast))
print("RMSE: ", np.roots(eval.mse(actual, forecast)))
print("CRPS: ", crps.calc_crps(forecast, actual))

prediction.plot(output_file='plots/graph.png')

actual, lower, upper = confidence.mean_confidence_interval(actual, 0.90)
compare_df = pd.DataFrame({'Actual': actual, 'Upper': upper, 'lower': lower, 'Forecast': forecast})

# plot the two vectors
ax = compare_df.plot(colormap='jet', marker='.', markersize=10, title='Forecasting Electricity Consumption')
ax.set_xlabel("Time frequency")
ax.set_ylabel("Electricity consumption")
fig = ax.get_figure()
fig.savefig("plots/compare.png")

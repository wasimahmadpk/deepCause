import pandas as pd
import numpy as np


from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

from gluonts.distribution.multivariate_gaussian import MultivariateGaussianOutput

from gluonts.evaluation.backtest import make_evaluation_predictions
    
        
# Generate data 

N = 20  # number of time series
T = 1000  # number of timesteps
dim = 2 # dimension of the observations
prediction_length = 25
freq = '1H'

custom_datasetx = np.random.normal(size=(N, dim, T))
custom_datasetx[:,1,:] = 5*custom_datasetx[:,1,:]
start = pd.Timestamp("01-01-2019", freq=freq)

train_ds = ListDataset(
    [
        {'target': x, 'start': start}
        for x in custom_datasetx[:, :, :-prediction_length]
    ],
    freq=freq,
    one_dim_target=False,
)



test_ds = ListDataset(
    [
        {'target': x, 'start': start}
        for x in custom_datasetx[:, :, :]
    ],
    freq=freq,
    one_dim_target=False,
)

# Deep AR 

# Trainer parameters
epochs = 10
learning_rate = 1E-3
batch_size = 5
num_batches_per_epoch = 100

# create estimator
estimator = DeepAREstimator(
    prediction_length=prediction_length,
    context_length=prediction_length,
    freq=freq,
    trainer=Trainer(
        ctx="cpu",
        epochs=epochs,
        learning_rate=learning_rate,
        hybridize=True,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
    ),
    distr_output=MultivariateGaussianOutput(dim=dim)
)

predictor = estimator.train(train_ds)


forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)


forecasts = list(forecast_it)
tss = list(ts_it)

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot') # this was just used for the examples

# data
t = np.linspace(50, 100, 100)
y = 5 * np.sin(t / 10) + 4 * np.random.randn(100 * 150).reshape(150, 100)
y_ = 5 * np.sin(t / 10) + 4 * np.random.randn(100 * 4000).reshape(4000, 100)

t2 = np.linspace(0, 50, 50)
y2 = 5 * np.sin(t2 / 10) + 4 * np.random.randn(50 * 150).reshape(150, 50)
y_2 = 5 * np.sin(t / 10) + 4 * np.random.randn(100 * 4000).reshape(4000, 100)


t__ = np.linspace(0, 100, 6)
y__ = 5 * np.sin(t__ / 10) + 4 * np.random.randn(6 * 4000).reshape(4000, 6)


def forecastPlot(x, y, n=20, percentile_min=1, percentile_max=99, color='r', plot_mean=True, plot_median=False,
           line_color='k', **kwargs):
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0)
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if 'alpha' in kwargs:
        alpha = kwargs.pop('alpha')
    else:
        alpha = 1 / n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        plt.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None, label=f'{percentile_max}% Confidence Interval')

    if plot_mean:
        if y.ndim == 1:
            y = y
        else:
            y = np.mean(y, axis=0)

        plt.plot(x, y, color=line_color, label='Prediction Mean')
        plt.legend("Prediction Mean", f"{percentile_max}% Confidence Interval")

    if plot_median:
        if y.ndim == 1:
            y = y
        else:
            y = np.median(y, axis=0)

        plt.plot(x, y, color=line_color, label='Prediction Median')
        plt.legend()

    return plt.gca()


# forecastPlot(t, y_, n=1, percentile_min=25, percentile_max=75, plot_median=False, plot_mean=False, color='b', line_color='navy', alpha=0.3)
# forecastPlot(t, y_, n=1, percentile_min=5, percentile_max=95, plot_median=True, plot_mean=False, color='g', line_color='navy', alpha=0.3)
# plt.plot(t2, np.median(y2, axis=0))

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fn_out = 'peak.png'

colors = np.array(["#ff0000", "#ff7777", "#ff00ff", "#ff77ff", "#00ff00", "#77ff77", "#0000ff", "#7777ff"])
custom_cmap = ListedColormap(colors)

values = []

fn_in_0 = 'pTCR.csv'

with open(fn_in_0, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    headers = next(reader)
    data_0 = np.array(list(reader))

ticklabels = headers

data_for_calc_0 = data_0.astype(float)
values_0 = np.mean(data_for_calc_0, 0)

fn_in = f"results_fit10_peak.csv"

with open(fn_in, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    headers = next(reader)
    data = np.array(list(reader))

data[np.where(data == '')] = 'nan'
data_for_calc = data[:, 1:].astype(float)
print(data_for_calc.shape)
means = np.nanmean(data_for_calc, 0)
logmeans = np.log10(means)
values.append(logmeans)

fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.scatter(values[0], values[1], values[2], color=colors)
# ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

ax = fig.add_subplot(1, 1, 1)
ax.set_title('peak')
sc = ax.scatter(values_0, values[0], c=range(8), cmap=custom_cmap, s=5, marker='.')
ax.scatter(values_0[[0, 2, 4, 6]], values[0][[0, 2, 4, 6]], c=[0, 2, 4, 6], cmap=custom_cmap, s=75, marker='o')
ax.scatter(values_0[[1, 3, 5, 7]], values[0][[1, 3, 5, 7]], c=[1, 3, 5, 7], cmap=custom_cmap, s=75, marker='^')
ax.set_xlabel('pTCR, MFI (arb.units)')
ax.set_ylabel(f'log10(peak)')
cbar = fig.colorbar(sc, boundaries=np.arange(9) - 0.5)
cbar.set_ticks(np.arange(8))
cbar.set_ticklabels(ticklabels)

plt.savefig(fn_out)

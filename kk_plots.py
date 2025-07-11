import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy import log10
import scipy.stats as stats

fn_out = 'kk_3dplot.png'

kNames1 = ["k1", "k2", "k3"]
kNames2 = ["kon", "koff", "krel"]

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

for i in range(3):
    kName = kNames1[i]
    kName2 = kNames2[i]

    fn_in = f"results_fit10_{kName}.csv"

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

fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(2, 2, 1, projection='3d')
# ax.scatter(values[0], values[1], values[2], color=colors)
# ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

ax = fig.add_subplot(4, 4, 4)
ax.set_title(kNames2[0])
sc = ax.scatter(values_0, values[0], c=range(8), cmap=custom_cmap, s=5, marker='.')
ax.scatter(values_0[[0, 2, 4, 6]], values[0][[0, 2, 4, 6]], c=[0, 2, 4, 6], cmap=custom_cmap, s=75, marker='o')
ax.scatter(values_0[[1, 3, 5, 7]], values[0][[1, 3, 5, 7]], c=[1, 3, 5, 7], cmap=custom_cmap, s=75, marker='^')
xx = [values_0[0], values_0[0], values_0[0]]
yy = [values[0][0] + log10(0.5), values[0][0] + log10(0.2), values[0][0] + log10(0.1)]
ax.scatter(xx, yy, facecolors='none', edgecolors="#ff0000", s=75, marker='o')
ax.set_xlabel('pTCR, MFI (arb.units)')
ax.set_ylabel(f'log10({kNames2[0]})')
# cbar = fig.colorbar(sc, boundaries=np.arange(9) - 0.5)
# cbar.set_ticks(np.arange(8))
# cbar.set_ticklabels(ticklabels)

ax = fig.add_subplot(4, 4, 13)
ax.set_title(kNames2[1])
sc = ax.scatter(values_0, values[1], c=range(8), cmap=custom_cmap, s=5, marker='.')
ax.scatter(values_0[[0, 2, 4, 6]], values[1][[0, 2, 4, 6]], color=colors[[0, 2, 4, 6]], s=75, marker='o')
ax.scatter(values_0[[1, 3, 5, 7]], values[1][[1, 3, 5, 7]], color=colors[[1, 3, 5, 7]], s=75, marker='^')
ax.set_xlabel('pTCR, MFI (arb.units)')
ax.set_ylabel(f'log10({kNames2[1]})')
# cbar = fig.colorbar(sc, boundaries=np.arange(9) - 0.5)
# cbar.set_ticks(np.arange(8))
# cbar.set_ticklabels(ticklabels)

ax = fig.add_subplot(4, 4, 16)
ax.set_title(kNames2[2])
sc = ax.scatter(values_0, values[2], c=range(8), cmap=custom_cmap, s=5, marker='.')
ax.scatter(values_0[[0, 2, 4, 6]], values[2][[0, 2, 4, 6]], color=colors[[0, 2, 4, 6]], s=75, marker='o')
ax.scatter(values_0[[1, 3, 5, 7]], values[2][[1, 3, 5, 7]], color=colors[[1, 3, 5, 7]], s=75, marker='^')
ax.set_xlabel('pTCR, MFI (arb.units)')
ax.set_ylabel(f'log10({kNames2[2]})')
# cbar = fig.colorbar(sc, boundaries=np.arange(9) - 0.5)
# cbar.set_ticks(np.arange(8))
# cbar.set_ticklabels(ticklabels)

plt.savefig(fn_out)

rho, p_value = stats.spearmanr(values_0, values[0])
print(f"kon Spearman’s ρ: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.spearmanr(values_0, values[1])
print(f"koff Spearman’s ρ: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.spearmanr(values_0, values[2])
print(f"krel Spearman’s ρ: {rho:.3f}, p-value: {p_value:.4f}")

rho, p_value = stats.pearsonr(values_0, values[0])
print(f"kon Pearson’s ρ: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.pearsonr(values_0, values[1])
print(f"koff Pearson’s ρ: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.pearsonr(values_0, values[2])
print(f"krel Pearson’s ρ: {rho:.3f}, p-value: {p_value:.4f}")

rho, p_value = stats.kendalltau(values_0, values[0])
print(f"kon kendall tau: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.kendalltau(values_0, values[1])
print(f"koff kendall tau: {rho:.3f}, p-value: {p_value:.4f}")
rho, p_value = stats.kendalltau(values_0, values[2])
print(f"krel kendall tau: {rho:.3f}, p-value: {p_value:.4f}")
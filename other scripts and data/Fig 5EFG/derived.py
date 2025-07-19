import os
import csv

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.integrate import odeint

fn_in = 'all_fitting_fit10.csv'

LL0 = [1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9]
LL1 = LL0 * 9
LL2 = []
L2 = 0.001
for i in range(9):
    for j in range(len(LL0)):
        LL2.append(L2)
    L2 *= 10
LL = np.array(LL1) * np.array(LL2)
print(LL)



def f(tt, L, A, B, k_p_1, k_p_2, k_p_3):
    yy = []

    k_m_1 = 0
    k_m_2 = 0
    k_m_3 = 0

    a11 = -(k_m_1 + k_p_2 + k_m_2 * L)
    a12 = k_p_1 * L - k_m_2 * L
    a21 = k_m_1 - k_p_3
    a22 = -(k_p_3 + k_m_3 + k_p_1 * L)

    b1 = k_m_2 * L
    b2 = k_p_3

    def g(y, t):
        y1 = y[0]
        y2 = y[1]
        dy1 = a11 * y1 + a12 * y2 + b1
        dy2 = a21 * y1 + a22 * y2 + b2
        return [dy1, dy2]

    g0 = [0, k_p_3 / (k_p_3 + k_m_3)]

    sol = odeint(g, g0, [0] + tt.tolist())
    sol = sol.tolist()
    sol.pop(0)
    for s in sol:
        X = s[0]
        R_a = s[1]

        y = B + A * X
        yy.append(y)

    return yy


line_number = 47

with open(fn_in, 'r') as file:
    reader = csv.reader(file, delimiter=';')
    headers = next(reader)
    data = np.array(list(reader))

fileName = str(data[line_number, 0]).replace("_fit10.txt", "")
ing = data[line_number, 1]
peptide = data[line_number, 3]
A = float(data[line_number, 4])
B = float(data[line_number, 5])
k_p_1 = float(data[line_number, 6])
k_p_2 = float(data[line_number, 7])
k_p_3 = float(data[line_number, 8])

print(fileName, ing, peptide, A, B, k_p_1, k_p_2, k_p_3)

logcoeff = 0

L0 = 10
tt = np.arange(0, 5.0, 0.1)
yy = f(tt, L0, A, B, k_p_1, k_p_2, k_p_3)
peak0 = np.max(yy)
print(peak0)

line_number = 44

fileName = str(data[line_number, 0]).replace("_fit10.txt", "")
ing = data[line_number, 1]
peptide = data[line_number, 3]
A = float(data[line_number, 4])
B = float(data[line_number, 5])
k_p_1 = float(data[line_number, 6])
k_p_2 = float(data[line_number, 7])
k_p_3 = float(data[line_number, 8])

print(fileName, ing, peptide, A, B, k_p_1, k_p_2, k_p_3)

line_number = 45

T4_A = float(data[line_number, 4])
T4_B = float(data[line_number, 5])
T4_k_p_1 = float(data[line_number, 6])
T4_k_p_2 = float(data[line_number, 7])
T4_k_p_3 = float(data[line_number, 8])

line_number = 46

V4_A = float(data[line_number, 4])
V4_B = float(data[line_number, 5])
V4_k_p_1 = float(data[line_number, 6])
V4_k_p_2 = float(data[line_number, 7])
V4_k_p_3 = float(data[line_number, 8])

line_number = 47

UVpep_A = float(data[line_number, 4])
UVpep_B = float(data[line_number, 5])
UVpep_k_p_1 = float(data[line_number, 6])
UVpep_k_p_2 = float(data[line_number, 7])
UVpep_k_p_3 = float(data[line_number, 8])

kons = []
koffs = []
peaks = []

blue = '#7777ff'
red = '#ff7777'

for coeff1 in LL:
    for coeff2 in LL:
        kons.append(coeff1)
        koffs.append(coeff2)
        yy = f(tt, L0, A, B, k_p_1 * coeff1, k_p_2 * coeff2, k_p_3)
        peak = np.max(yy)
        peaks.append(peak)
kons = np.log10(np.array([kons]))
koffs = np.log10(np.array([koffs]))
colors = np.array([(1 if x else 0) for x in (peaks > peak0)])
print(colors)
cmap = ListedColormap([blue, red])
plt.figure(figsize=(4.5, 4.5))
plt.scatter(kons, koffs, c=colors, cmap=cmap, s=20)
plt.scatter([0], [0], color='black', marker='*', s=400)
plt.scatter([np.log10(T4_k_p_1 / k_p_1)], [np.log10(T4_k_p_2 / k_p_2)], color='green', marker='*', s=400)
plt.scatter([np.log10(V4_k_p_1 / k_p_1)], [np.log10(V4_k_p_2 / k_p_2)], color='cyan', marker='*', s=400)
plt.scatter([np.log10(UVpep_k_p_1 / k_p_1)], [np.log10(UVpep_k_p_2 / k_p_2)], color='yellow', marker='*', s=400)
plt.xlabel('log10 ( kon / kon_N4 )')
plt.ylabel('log10 ( koff / koff_N4 )')
out_file = f"{fileName}_{peptide}_{ing}_kon_koff.png"
plt.savefig(out_file)

# koffs = []
# krels = []
# peaks = []
# for coeff1 in LL:
#     for coeff2 in LL:
#         koffs.append(coeff1)
#         krels.append(coeff2)
#         yy = f(tt, L0, A, B, k_p_1, k_p_2 * coeff1, k_p_3 * coeff2)
#         peak = np.max(yy)
#         peaks.append(peak)
# koffs = np.log10(np.array([koffs]))
# krels = np.log10(np.array([krels]))
# colors = np.array([(1 if x else 0) for x in (peaks > peak0)])
# print(colors)
# cmap = ListedColormap([blue, red])
# plt.figure(figsize=(4.5, 4.5))
# plt.scatter(koffs, krels, c=colors, cmap=cmap, s=200)
# plt.scatter([0], [0], color='black', marker='*', s=400)
# plt.scatter([np.log10(T4_k_p_2 / k_p_2)], [np.log10(T4_k_p_3 / k_p_3)], color='green', marker='*', s=400)
# plt.scatter([np.log10(V4_k_p_2 / k_p_2)], [np.log10(V4_k_p_3 / k_p_3)], color='cyan', marker='*', s=400)
# plt.scatter([np.log10(UVpep_k_p_2 / k_p_2)], [np.log10(UVpep_k_p_3 / k_p_3)], color='yellow', marker='*', s=400)
# plt.xlabel('log10 ( koff / koff_N4 )')
# plt.ylabel('log10 ( krel / krel_N4 )')
# out_file = f"{fileName}_{peptide}_{ing}_koff_krel.png"
# plt.savefig(out_file)
#
# krels = []
# kons = []
# peaks = []
# for coeff1 in LL:
#     for coeff2 in LL:
#         krels.append(coeff1)
#         kons.append(coeff2)
#         yy = f(tt, L0, A, B, k_p_1 * coeff2, k_p_2, k_p_3 * coeff1)
#         peak = np.max(yy)
#         peaks.append(peak)
# krels = np.log10(np.array([krels]))
# kons = np.log10(np.array([kons]))
# colors = np.array([(1 if x else 0) for x in (peaks > peak0)])
# print(colors)
# cmap = ListedColormap([blue, red])
# plt.figure(figsize=(4.5, 4.5))
# plt.scatter(krels, kons, c=colors, cmap=cmap, s=200)
# plt.scatter([0], [0], color='black', marker='*', s=400)
# plt.scatter([np.log10(T4_k_p_3 / k_p_3)], [np.log10(T4_k_p_1 / k_p_1)], color='green', marker='*', s=400)
# plt.scatter([np.log10(V4_k_p_3 / k_p_3)], [np.log10(V4_k_p_1 / k_p_1)], color='cyan', marker='*', s=400)
# plt.scatter([np.log10(UVpep_k_p_3 / k_p_3)], [np.log10(UVpep_k_p_1 / k_p_1)], color='yellow', marker='*', s=400)
# plt.xlabel('log10 ( krel / krel_N4 )')
# plt.ylabel('log10 ( kon / kon_N4 )')
# out_file = f"{fileName}_{peptide}_{ing}_krel_kon.png"
# plt.savefig(out_file)


# for k in range(6):
#     dirName = f"{fileName}_{peptide}_{ing}_coeff_1E{logcoeff}"
#     print(dirName)
#
#     os.makedirs(dirName, exist_ok=True)
#
#     coeff = 10 ** logcoeff
#     tt = np.arange(0, 32 / coeff, 0.1 / coeff)
#
#     i = 0
#     for L in LL:
#         i += 1
#         yy = f(tt, L, A, B, k_p_1 * coeff, k_p_2 * coeff, k_p_3 * coeff)
#         plt.figure()
#         plt.plot(tt, yy)
#         plt.title(f"L = {L}")
#         out_file = os.path.join(dirName, f"{i}_L{L}.png")
#         plt.savefig(out_file)
#
#     logcoeff -= 1

font = {'size': 32}

plt.rc('font', **font)

lw = 5

line_number = 44

fileName = str(data[line_number, 0]).replace("_fit10.txt", "")
ing = data[line_number, 1]
peptide = data[line_number, 3]
A = float(data[line_number, 4])
B = float(data[line_number, 5])
k_p_1 = float(data[line_number, 6])
k_p_2 = float(data[line_number, 7])
k_p_3 = float(data[line_number, 8])

out_file = f"{fileName}_{peptide}_{ing}_L.png"
fig = plt.figure(figsize=(14, 9))
tt = np.arange(0, 32, 0.1)
colors = ['red', 'green', 'blue', 'magenta', 'cyan']
cmap = ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(6) - 0.5, 5)
ticklabels = ['L = 0.5', 'L = 1.0', 'L = 2.0', 'L = 5.0', 'L = 10.0']
yy = f(tt, 0.5, A, B, k_p_1, k_p_2, k_p_3)
pl = plt.plot(tt, yy, color='red', linewidth=lw)
yy = f(tt, 1.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='green', linewidth=lw)
yy = f(tt, 2.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='blue', linewidth=lw)
yy = f(tt, 5.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='magenta', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='cyan', linewidth=lw)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), boundaries=np.arange(6) - 0.5)
cbar.set_ticks(np.arange(5))
cbar.set_ticklabels(ticklabels)
plt.xlabel('T, min')
plt.ylabel('TCR-pMHC, MFI (arb. units)')
plt.ylim([0, 2800])
plt.savefig(out_file)

line_number = 45

fileName = str(data[line_number, 0]).replace("_fit10.txt", "")
ing = data[line_number, 1]
peptide = data[line_number, 3]
A = float(data[line_number, 4])
B = float(data[line_number, 5])
k_p_1 = float(data[line_number, 6])
k_p_2 = float(data[line_number, 7])
k_p_3 = float(data[line_number, 8])

out_file = f"{fileName}_{peptide}_{ing}_L.png"
fig = plt.figure(figsize=(14, 9))
tt = np.arange(0, 32, 0.1)
colors = ['green', 'cyan', 'black', 'brown']
cmap = ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(5) - 0.5, 5)
ticklabels = ['L = 1.0', 'L = 10.0', 'L = 50.0', 'L = 100.0']
yy = f(tt, 1.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='green', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='cyan', linewidth=lw)
yy = f(tt, 50.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='black', linewidth=lw)
yy = f(tt, 100.0, A, B, k_p_1, k_p_2, k_p_3)
plt.plot(tt, yy, color='brown', linewidth=lw)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), boundaries=np.arange(5) - 0.5)
cbar.set_ticks(np.arange(4))
cbar.set_ticklabels(ticklabels)
plt.xlabel('T, min')
plt.ylabel('TCR-pMHC, MFI (arb. units)')
plt.ylim([0, 2800])
plt.savefig(out_file)

line_number = 44

fileName = str(data[line_number, 0]).replace("_fit10.txt", "")
ing = data[line_number, 1]
peptide = data[line_number, 3]
A = float(data[line_number, 4])
B = float(data[line_number, 5])
k_p_1 = float(data[line_number, 6])
k_p_2 = float(data[line_number, 7])
k_p_3 = float(data[line_number, 8])

out_file = f"{fileName}_{peptide}_{ing}_kon.png"
fig = plt.figure(figsize=(14, 9))
tt = np.arange(0, 32, 0.1)
colors = ['red', 'green', 'blue', 'magenta']
cmap = ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(5) - 0.5, 5)
ticklabels = ['kon * 1.0', 'kon * 0.5', 'kon * 0.2', 'kon * 0.1']
yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3)
pl = plt.plot(tt, yy, color='red', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1 * 0.5, k_p_2, k_p_3)
plt.plot(tt, yy, color='green', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1 * 0.2, k_p_2, k_p_3)
plt.plot(tt, yy, color='blue', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1 * 0.1, k_p_2, k_p_3)
plt.plot(tt, yy, color='magenta', linewidth=lw)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), boundaries=np.arange(5) - 0.5)
cbar.set_ticks(np.arange(4))
cbar.set_ticklabels(ticklabels)
plt.xlabel('T, min')
plt.ylabel('TCR-pMHC, MFI (arb. units)')
plt.savefig(out_file)

out_file = f"{fileName}_{peptide}_{ing}_koff.png"
fig = plt.figure(figsize=(14, 9))
tt = np.arange(0, 32, 0.1)
colors = ['red', 'green', 'blue', 'magenta']
cmap = ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(5) - 0.5, 5)
ticklabels = ['koff * 1.0', 'koff * 2.0', 'koff * 5.0', 'koff * 10.0']
yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3)
pl = plt.plot(tt, yy, color='red', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1, k_p_2 * 2.0, k_p_3)
plt.plot(tt, yy, color='green', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1, k_p_2 * 5.0, k_p_3)
plt.plot(tt, yy, color='blue', linewidth=lw)
yy = f(tt, 10.0, A, B, k_p_1, k_p_2 * 10.0, k_p_3)
plt.plot(tt, yy, color='magenta', linewidth=lw)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), boundaries=np.arange(5) - 0.5)
cbar.set_ticks(np.arange(4))
cbar.set_ticklabels(ticklabels)
plt.xlabel('T, min')
plt.ylabel('TCR-pMHC, MFI (arb. units)')
plt.savefig(out_file)

# out_file = f"{fileName}_{peptide}_{ing}_krel.png"
# fig = plt.figure(figsize=(14, 9))
# tt = np.arange(0, 32, 0.1)
# colors = ['red', 'green', 'blue', 'magenta']
# cmap = ListedColormap(colors)
# norm = matplotlib.colors.BoundaryNorm(np.arange(5) - 0.5, 5)
# ticklabels = ['krel * 1.0', 'krel * 2.0', 'krel * 5.0', 'krel * 10.0']
# yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3)
# pl = plt.plot(tt, yy, color='red', linewidth=lw)
# yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3 * 2.0)
# plt.plot(tt, yy, color='green', linewidth=lw)
# yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3 * 5.0)
# plt.plot(tt, yy, color='blue', linewidth=lw)
# yy = f(tt, 10.0, A, B, k_p_1, k_p_2, k_p_3 * 10.0)
# plt.plot(tt, yy, color='magenta', linewidth=lw)
# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=plt.gca(), boundaries=np.arange(5) - 0.5)
# cbar.set_ticks(np.arange(4))
# cbar.set_ticklabels(ticklabels)
# plt.xlabel('T, min')
# plt.ylabel('TCR-pMHC, MFI (arb. units)')
# plt.savefig(out_file)
## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (8,8) 
import numpy as np


delta_phi = np.array([0.51815415, 0.5 + 0.5*0.03804, 0.5 + 0.5*0.03646, 0.5 + 0.5*0.036012])
factor = 4**delta_phi
factormu = 4**(delta_phi[1:]-2)
a_s2 = np.array([6.7537, 8.568, 10.075, 11.61527])
A_s2 = np.divide(a_s2, factor)
# print(A_s2)
Ncont1 = np.linspace(1, 5, 100)
Ncont2 = np.linspace(0, 5, 100)
A_cont = Ncont2 + 0.678
N = [1,2,3,4]
Nmu = [2,3,4]
mu_phi2 = [0.2394, 0.2539, 0.26337]
mu_phi2 = np.divide(mu_phi2, factormu)
mu_phicont = 2*(1 + np.divide(0.678, Ncont1))

plt.plot(Ncont2, A_cont)
plt.plot(N, A_s2, 'o', color='black')
# plt.plot(Ncont1, mu_phicont)
# plt.plot(Nmu, mu_phi2, 'o', color='black')
plt.xlabel("$N$")
plt.ylabel("$$\mu_\phi^2$$")
plt.show()
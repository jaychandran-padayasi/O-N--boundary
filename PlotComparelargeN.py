## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (8,8) 
import numpy as np


delta_phi = np.array([0.518149, 0.519088, 0.51928, 0.518, 0.5169, 0.51155, 0.5064])
factor_as2 = 4**delta_phi
factor_mu = 4**(delta_phi[1:]-2)
a_s2 = np.array([6.756403386923523, 8.510808577776043, 10.160577453220649, 11.617150209315772, 13.258463115630398, 22.219464503524854, 41.353512420744764])
a_s2sym = np.array([6.756403386923523, 7.257002932709902, 7.781005560682007, 8.564836581550919, 9.311459295666884, 11.127304873438051, 11.8292486633929])
A_s2 = np.divide(a_s2, factor_as2)
A_s2sym = np.divide(a_s2sym, factor_as2)

# print(A_s2)
Ncont1 = np.linspace(1, 20.5, 100)
Ncont2 = np.linspace(0, 20.5, 100)
A_cont = Ncont2 + 0.678
N = [1, 2, 3, 4, 5, 10, 20]

mu_phi2 = [0.24129642689422473, 0.2509752396438516, 0.26333670354655636, 0.2669765352654295, 0.2690704968153803, 0.2650142345860297]
mu_phi2 = np.divide(mu_phi2, factor_mu)
mu_phicont = 2*(1 + np.divide(0.678, Ncont1))

Alphas = []
for i in range(1, len(N)):
    alpha = (A_s2[i]/mu_phi2[i-1]) - (N[i] - 2)
    alpha /= 2*np.pi
    Alphas.append(alpha)
print(Alphas)
plt.plot(Ncont2, A_cont, color='blue', label='large-N')
plt.plot(N, A_s2, 'o', color='black', label='Bootstrap')
plt.plot(N, A_s2sym, 'o', color='red', label='Bootstrap (alternate)')

# plt.plot(Ncont1, mu_phicont, color='blue', label='large-N')
# plt.plot(N[1:], mu_phi2, 'o', color='black', label='Bootstrap (current work)')

# plt.plot(N[1:], Alphas,  '--o', color='red')
# plt.hlines(0, 1.9, 20.1, colors='black', linestyles='solid')

plt.xticks(ticks=N)
plt.xlabel("$N$")
plt.legend()
plt.ylabel("$$A_{\sigma}^2 $$")
plt.show()
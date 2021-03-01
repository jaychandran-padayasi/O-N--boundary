## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (8,8) 
import numpy as np


del_e2 = [14.021557617187508, 13.401489257812504, 13.085156250000002, 12.911132812500002, 12.9671630859375, 13.09929199, 13.06970215, 12.88952637, 12.75959473, 12.70056152, 12.6501709]
A_s2 = [8.715779960659102, 8.686691522049008, 8.653416539425624, 8.627492950566715, 8.654995040256829, 8.699697318615891, 8.71703969858452, 8.66786659457212, 8.627841716551583, 8.607605480418114, 8.588860963968525]
mu_phi2 = [0.21912221494942416, 0.22221829166304902, 0.22578573160213097, 0.22857323880783256, 0.22552594205026005, 0.22059760280174587, 0.218590903416384, 0.22396191590285502, 0.2283400604840663, 0.2305595568571927, 0.232621287047525]
mu_phi2 = [0.24129642689422473/muphi for muphi in mu_phi2]
del_e2  = [del2/12.48153781 for del2 in del_e2]
A_s2 = [a/8.510808577776043 for a in A_s2]

del_e3 = [16, 18, 20, 22, 24, 25,30, 35, 43, 50, 60]

del_e2_bdy = [11.9337158203125, 12.048925781249999, 12.126562500000002, 12.15622559, 12.25305176, 12.30593262, 12.33911133, 12.36174316, 12.39074707]
A_s2_bdy = [7.673762973369475, 8.01438388303278, 8.164233086009453, 8.210694018580696, 8.332911731002344, 8.385126418450339, 8.413862645296485, 8.431936797071739, 8.453455535871646]
mu_phi2_bdy = [0.3502148919242186, 0.30178097785453, 0.28242012446853243, 0.276626727743986, 0.26182184383752927, 0.25567800578479083, 0.25234046692450546, 0.2502565850788457, 0.24779079127552342]
mu_phi2_bdy = [0.24129642689422473/muphi for muphi in mu_phi2_bdy]
A_s2_bdy = [a/8.510808577776043 for a in A_s2_bdy]
del_e2_bdy = [del2/12.48153781 for del2 in del_e2_bdy]
del_O = [5,7,9, 10, 15, 20, 25, 30, 40]

# plt.plot(del_O, del_e2_bdy, '--o', color='black', label='$$\Delta_{\epsilon^{\prime\prime\prime}}/\Delta^{(0)}_{\epsilon^{\prime\prime\prime}}$$')
# plt.plot(del_O, A_s2_bdy, '--x', color='red', label='$$a_{\sigma}^2/a_{\sigma}^{2(0)}$$' )
# plt.plot(del_O, mu_phi2_bdy, '--v', color='blue', label='$$\mu_{\phi\phi}^{2(0)}/\mu_{\phi\phi}^2$$' )
# plt.xticks(del_O)


plt.plot(del_e3, del_e2, '--o', color='black', label='$$\Delta_{\epsilon^{\prime\prime\prime}}/\Delta^{(0)}_{\epsilon^{\prime\prime\prime}}$$')
plt.plot(del_e3, A_s2, '--x', color='red', label='$$a_{\sigma}^2/a_{\sigma}^{2(0)}$$')
plt.plot(del_e3, mu_phi2, '--v', color='blue', label='$$\mu_{\phi\phi}^{2(0)}/\mu_{\phi\phi}^2$$')
plt.xticks(del_e3)

plt.legend()
plt.xlabel("$$\hat{\Delta}_{\epsilon^4}$$")
plt.ylabel("Relative change in CFT data")
plt.show()
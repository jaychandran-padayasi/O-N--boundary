## Bootstrap imports
from sympy import diff
from sympy.functions import hyper
from sympy.abc import symbols
import numpy as np
from scipy.linalg import svdvals
import scipy.special as sc
from scipy.optimize import linprog
import time

## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (14,8) 

np.set_printoptions(precision=7)

dim = 3
M_max = 15
num_boundary = 30
num_bulk = 30
bulk_begin, bulk_end = 1.0, 10.0
bdy_begin, bdy_end = 2.9, 7.0


delta_1 = 0.51928
delta_2 = 0.51928
delta_12 = 0

class ConformalBlockTable:
		"""
		Generates a table of Boundary Conformal Blocks for two-point functions
		and their derivatives. 

		Attributes
		----------
		table :   The block table consisting of sympy expressions of all the bulk 
				  and boundary conformal blocks. table[0,:] contains all the bulk
				  blocks (expansions of 2-point function of B-CFT in bulk operators). 
				  Call a particular derivative by table[1,2] (second derivative of 
				  expansion in boundary operators)

		m_order : list of derivatives, just to remind you that the original block
				  is still returned, along with the derivatives
		
		dim :     dimension of the table
		"""
		def __init__(self, dim, M_max):
				self.dim = dim
				self.m_order = list(range(M_max+1))
				Delta, Delta_12, Xi = symbols('Delta Delta_12 Xi')
				Bulk_expr = -1*(Xi**(Delta/2))*hyper([(Delta + Delta_12)/2 , (Delta - Delta_12)/2], [Delta + 1 - (dim/2)],-Xi)
				Bdy_expr = (Xi**(-Delta + (delta_1 + delta_2)/2))*hyper([Delta, Delta + 1 - (dim/2)], [2*Delta + 2 - (dim)], -Xi**(-1))
				# if dim == 3:
				#         Bdy_expr = (0.5/(Xi**0.5))*((4/(1 + Xi))**(Delta - 0.5))*(1 + (Xi/(1+Xi))**(0.5))**(-2*(Delta - 1))
				Bulk_array = [Bulk_expr]
				Bdy_array = [Bdy_expr]
				for n in range(1, M_max + 1):
					# print('Differentiating n = ', n)
					Bulk_array.append(diff(Bulk_expr, Xi, n))
					Bdy_array.append(diff(Bdy_expr, Xi, n))
				self.table = np.array([Bulk_array, Bdy_array])      
		
		def bulk_block(self, delta):
				block_array = []
				for n in range(M_max+1):
					eval = self.table[0,n].subs({"Delta":delta, "Delta_12":0.00, "Xi":1.00}).evalf()
					block_array.append(eval)
				return block_array
		
		def bdy_block(self, delta):
				bdy_array = []
				for n in range(M_max+1):
					eval = -1*self.table[1,n].subs({"Delta":delta, "Delta_12":0.00, "Xi":1.00}).evalf()
					bdy_array.append(eval)
				return bdy_array

		def identity_block(self):
				identity_array = [-1]
				Xi = symbols("Xi")
				identity_expr = -1*Xi**((delta_1 + delta_2)/2)
				for n in range(1, M_max+1):
					eval = diff(identity_expr, Xi, n).subs({"Xi":1.00}).evalf()
					identity_array.append(eval)
				return identity_array
					

print('Getting Conformal block table')
start_time = time.time()
table1 = ConformalBlockTable(dim, M_max)
end_time = time.time() - start_time
print('Done, took', np.round(end_time, 2), 'sec')

bulk_dims = np.linspace(bulk_begin, bulk_end, num_bulk)
bdy_dims = np.linspace(bdy_begin, bdy_end, num_boundary)

print('Constructing F_\Delta')
start_time = time.time()
Eff_Delta = []
for bulkOp in bulk_dims:
	Eff_Delta.append(table1.bulk_block(bulkOp))

Eff_Delta.append(table1.identity_block())

for bdyOp in bdy_dims:
	Eff_Delta.append(table1.bdy_block(bdyOp))

Eff_Delta = np.array(Eff_Delta)
print(np.shape(Eff_Delta))
end_time = time.time() - start_time
print('Done, took', np.round(end_time, 2), 'sec')


c_T = np.array([1] + [0]*M_max)
b_ub = np.array([0]*(num_boundary + num_bulk + 1))
b_eq = 1
A_eq = np.array([table1.bdy_block(dim-1)])

res = linprog(c_T, Eff_Delta, b_ub, A_eq, b_eq, method='revised simplex', options={"disp":True, "maxiter":4000})
print(res)
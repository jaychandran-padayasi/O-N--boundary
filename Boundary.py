## Bootstrap imports
from sympy import diff
from sympy.functions import hyper
from sympy.abc import symbols
import numpy as np
from scipy.linalg import svdvals
import scipy.special as sc
import time

## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (8,8)


#################################################
# DEFINE THE FOLLOWING BEFORE RUNNING EVERYTIME #
#################################################

dim = 3
M_max = 5

# Operator list, only works for scalar blocks 
Bulk_Operators = []
Bdy_Operators = []

# # Specify External Dimensions
delta_1 = 0
delta_2 = 0
delta_12 = delta_1 - delta_2

# Customizable evaluate() function for unknowns given a matrix
# of SymPy expressions
def evaluate(Matrix, variables):
		mat = np.zeros_like(Matrix, dtype=np.double)
		for index, expr in np.ndenumerate(Matrix):
				mat[index] = expr.subs({'Delta_phi':delta_phi, 'Delta_4':delta_4}).evalf()
		return mat

####################################################
# CLASS AND FUNCTION DEFINITIONS (INTERNAL)        #
# MEANT TO BE FOLLOWED BY CLIENT CODE TO DETERMINE #
# VANISHING SINGULAR VALUE                         #
####################################################

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
				Bdy_expr = (Xi**(-Delta))*hyper([Delta, Delta + 1 - (dim/2)], [2*Delta + 2 - (dim)], -Xi**(-1))
				Bulk_array = [Bulk_expr]
				Bdy_array = [Bdy_expr]
				for n in range(1, M_max + 1):
						Bulk_array.append(diff(Bulk_expr, Xi, n))
						Bdy_array.append(diff(Bdy_expr, Xi, n))
				self.table = np.array([Bulk_array, Bdy_array])      

table1 = ConformalBlockTable(dim, M_max)

def derMatrix(Bulk_Operators, Bdy_Operators, s=1):
		"""
		Generates a matrix in the same vein as Eq (2.10) of 1502.07217
		First n_bulk elements are coefficients of p_k's
		Next n_bdy elements are coefficients of q_l's
		Last element is coefficient of a_1*a_2

		Note that this matrix contains a mix of expressions and numbers

		FOLLOWING VARIABLES MUST BE DEFINED GLOBALLY BEFORE CALLING:
		table1, delta_1, delta_2, delta_12
		"""
		Matrix = []
		for i in range(1, M_max + 1):
				row = []
				for Bulk_Op in Bulk_Operators:
						table_call = table1.table[0, i].subs({"Delta":Bulk_Op, "Delta_12":delta_12, "Xi":1.00})
						row.append(table_call)
				for Bdy_Op in Bdy_Operators:
						table_call = table1.table[1, i].subs({"Delta":Bdy_Op, "Xi":1.00})
						row.append(table_call)
				Xi = symbols('Xi')
				if s == 1:
						last_expr = Xi**((delta_1 + delta_2)/2)
						last_elem = diff(last_expr, Xi, i).subs({"Xi":1.00}).evalf()
						row.append(last_elem)
				Matrix.append(row)
		return np.array(Matrix)


def inhomoCoeffs(Bulk_Operators, Bdy_Operators, s=1):
		"""
		Generates the first equation as Eq (2.10) of 1502.07217
		This is needed to calculate CFT data once dimensions
		are found.
		First n_bulk elements are coefficients of p_k's
		Next n_bdy elements are coefficients of q_l's
		Last element is coefficient of a_1*a_2

		FOLLOWING VARIABLES MUST BE DEFINED GLOBALLY BEFORE CALLING:
		table1, delta_1, delta_2, delta_12
		"""
		Row = []
		for Bulk_Op in Bulk_Operators:
				table_call = table1.table[0, 0].subs({"Delta":Bulk_Op, "Delta_12":delta_12, "Xi":1.00})
				Row.append(table_call)
		for Bdy_Op in Bdy_Operators:
				table_call = table1.table[1, 0].subs({"Delta":Bdy_Op, "Xi":1.00})
				Row.append(table_call)
		if s==1:
			last_expr = Xi**((delta_1 + delta_2)/2)
			last_elem = last_expr.subs({"Xi":1.00}).evalf()
			Row.append(last_elem)
			
		return np.array(Row)



###########################################
# FIND VANISHING SINGULAR VALUE           #
###########################################

table1 = ConformalBlockTable(dim, M_max)
derMatrix_expr = derMatrix(Bulk_Operators, Bdy_Operators)

## Example code for Ising model (One variable)

# dO_num = 200
# dO_range = np.linspace(1, 1.4, dO_num)
# logz_data = np.zeros(dO_num)

# for dO_index, dO in np.ndenumerate(dO_range):
#         tempmat = evaluate(derMatrix_expr, dO)
#         log_z_data[dO_index] = np.log(min(svdvals(tempmat)))

# plt.plot(dO_range, log_z_data)

# plt.xlabel('$$\Delta_{O}$$')
# plt.ylabel('log($z$)')
# plt.title('Log(min(singular value)) of derivative matrix$')
# plt.savefig('3d_Ising_Bdy.pdf', format='pdf')
# plt.show()



		





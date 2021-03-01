## Bootstrap imports
from sympy import diff
from sympy.functions import hyper
from sympy.abc import symbols
import numpy as np
from scipy.linalg import svdvals
import scipy.special as sc
from scipy.optimize import minimize
import time

## Plotting imports
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams['figure.figsize'] = (14,8) 

# np.set_printoptions(precision=7)

#################################################
# DEFINE THE FOLLOWING BEFORE RUNNING EVERYTIME #
#################################################

dim = 3
M_max = 8
N = 3
# Operator list, only works for scalar blocks 
eta = 0.0128
nu = 0.9416
omega = 0.887
Bulk_Operators = [3 - (1/nu), 3 + omega, 'Delta_e1', 'Delta_e2', 16, 18]
Bulk_Operators = [1.5957, 3.79101, 'Delta_e1', 'Delta_e2']
Bdy_Operators = [2.0, 3.0]



# # Specify External Dimensions
delta_1 = 0.5 + 0.5*eta
delta_2 = 0.5 + 0.5*eta
delta_1 = 0.51928
delta_2 = 0.51928
delta_12 = 0

print("N = ", N)
print('Bulk Operators: ', Bulk_Operators)
print("External Dimension: ", delta_1)

# Customizable evaluate() function for unknowns given a matrix
# of SymPy expressions
def evaluate(x, Matrix):
	delta_e1 = x[0]
	delta_e2 = x[1]
	mat = np.zeros_like(Matrix, dtype=np.double)
	for index, expr in np.ndenumerate(Matrix):
		mat[index] = float(expr.subs({'Delta_e1':delta_e1, 'Delta_e2':delta_e2}).evalf())
	return np.log(min(svdvals(mat)))

def evaluateMat(Matrix, delta_e1, delta_e2):
	mat = np.zeros_like(Matrix, dtype=np.double)
	for index, expr in np.ndenumerate(Matrix):
		mat[index] = float(expr.subs({'Delta_e1':delta_e1, 'Delta_e2':delta_e2}).evalf())
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
				Bdy_expr = (Xi**(-Delta + (delta_1 + delta_2)/2))*hyper([Delta, Delta + 1 - (dim/2)], [2*Delta + 2 - (dim)], -Xi**(-1))
				# if dim == 3:
				#         Bdy_expr = (0.5/(Xi**0.5))*((4/(1 + Xi))**(Delta - 0.5))*(1 + (Xi/(1+Xi))**(0.5))**(-2*(Delta - 1))
				Bulk_array = [Bulk_expr]
				Bdy_array = [Bdy_expr]
				for n in range(1, M_max + 1):
						Bulk_array.append(diff(Bulk_expr, Xi, n))
						Bdy_array.append(diff(Bdy_expr, Xi, n))
				self.table = np.array([Bulk_array, Bdy_array])      


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
			Xi = symbols('Xi')
			last_expr = Xi**((delta_1 + delta_2)/2)
			last_elem = last_expr.subs({"Xi":1.00}).evalf()
			Row.append(last_elem)
			
		return np.array(Row)



###########################################
# FIND VANISHING SINGULAR VALUE           #
###########################################
print('Getting Conformal Block Table')
table1 = ConformalBlockTable(dim, M_max)
derMatrix_expr = derMatrix(Bulk_Operators, Bdy_Operators)
print('Done.')


# tol = 1e-50
# delta_e2_num = 8
# delta_e1_num = 200

# delta_e1_range = np.linspace(11,13.5, delta_e1_num)
# delta_e2_range = np.linspace(30,100, delta_e2_num)

# log_zdata = np.zeros((delta_e2_num, delta_e1_num))

print('Minimizing Singular Values ...')
start_time = time.time()

res = minimize(evaluate, np.array([5,10]), args=(derMatrix_expr), method='Nelder-Mead', options={'disp':True})
print(res)
[delta_e1_min, delta_e2_min] = res.x
end_time = time.time() - start_time
print('Done, took ', np.round(end_time, 3), 'sec')

###########################################
# FIND CFT DATA                           #
###########################################

# Naive Strategy: Make a square matrix by ignoring extra constraints
# and ask Numpy to solve it.
inhomoRow_expr = inhomoCoeffs(Bulk_Operators, Bdy_Operators)
inhomoRow = []
for element in inhomoRow_expr:
	inhomoRow.append(float(element.subs({"Delta_e1":delta_e1_min, "Delta_e2":delta_e2_min}).evalf()))

inhomoRow = np.array(inhomoRow)
row_len = len(inhomoRow)
inhomoRow = np.reshape(inhomoRow, (1, len(inhomoRow)))
homoMat = evaluateMat(derMatrix_expr, delta_e1_min, delta_e2_min)
homoMat = homoMat[:row_len-1]
SqMat = np.concatenate((inhomoRow, homoMat), axis=0)
RHS = np.zeros(row_len)
RHS[0] = N
params = np.linalg.solve(SqMat, RHS)
print("Bulk OPE Coefficients:")
print("a_e*lambda_(sse) = {", params[:4]/N, "}")
print("Boundary OPE coefficients:")
print("mu_phi^2 = ", params[4]*(1/(N-1)))
print("mu_(sD)^2 = ", params[5])
print("a_s^2 = ", params[6])

###########################################
# PLOTS                                   #
###########################################

# for i in range(len(delta_e2_range)):
# 	plt.plot(delta_e1_range, log_zdata[i,:], label=str(np.round(delta_e2_range[i], 3)))


# plt.xlabel('$$\Delta_{\epsilon\'\'}$$')
# plt.ylabel('log(z)')
# mintext = '$$\Delta_O = ' + str(np.round(dO_range[ind], 3)) + "$$"
# ax = plt.gca()
# plt.text(0.05,0.8, mintext, horizontalalignment='left', verticalalignment='top', transform = ax.transAxes)

# plt.legend()
# # plt.savefig('Normal_N4.pdf', format='pdf')
# plt.show()








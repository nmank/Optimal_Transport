'''
Optimal transport example/visualization tool. 
The problem is formulated as a linear program then solved using the scipy.optimize linprog.

Input: locations of the supports of the two distributions along with their probabilities. 
(Set vis = False if you don't want a plot of the result.)

Output: The cost tr(D^T T) and the transport matrix. A matlab figure for your example should also appear.

By Nate Mankovich 2019
'''

import numpy as np 
import scipy.optimize as opt
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
from mod import Mod
import random
import math
from scipy.spatial.distance import cdist, euclidean


#user parameters
#location of the support of distrtibution 0 and distribution 1.
# l0 = np.array([[0,0,0,0],[0,.5,.75,1]])
# l1 = np.array([[.25,.5,.75],[1,1,1]])
# l2 = np.array([[.25,.75],[0,0]])
# #probabilities for each point in the support of distriobution 0 and distribution 1
# b0 = np.array([1/4,1/4,1/4,1/4])
# b1 = np.array([1/3,1/3,1/3])
# b2 = np.array([1/2,1/2])

#FIX
# #another example
# r=7
# s=7
# t=7
# #r+s+t = num_samples

# # make curves
# x1 = np.linspace(.1, 1, r)
# x2 = np.linspace(.1, 1, s)
# x3 = np.linspace(.1, 1, t)

# l0 = np.zeros((2,r))
# l1 = np.zeros((2,s))
# l2 = np.zeros((2,t))

# for i in range(r):
# 	l0[0,i] = x1[i]+.1
# 	l0[1,i] = 1-x1[i]+.1
# for i in range(s):
# 	l1[0,i] = x2[i]
# for i in range(t):
# 	l2[1,i] = x3[i]

# #b has all the same probabilities
# b = np.ones(num_samples)
# b0 = b[0:r]/r
# b1 = b[r:r+s]/s
# b2 = b[r+s:r+s+t]/t



#circle points
num_samples = 90
r=30
s=30
t=30
#r+s+t = num_samples
radius = 1

# make a simple unit circle 
theta = np.linspace(2*np.pi/num_samples, 2*np.pi, num_samples)

l = np.zeros((2,num_samples))
for i in range(num_samples):
	l[0,i] = radius*np.cos(theta[i])
	l[1,i] = radius*np.sin(theta[i])

l0 = l[:,0:r]
l1 = l[:,r:r+s]
l2 = l[:,r+s:r+s+t]

#b has all the same probabilities
b = np.ones(num_samples)
b0 = b[0:r]/r
b1 = b[r:r+s]/s
b2 = b[r+s:r+s+t]/t


#triangle points
#FIX
# r=5
# s=r
# t=r

# sp = np.linspace(1/num_samples, .1, num_samples)

def run_example_triangles(l0,l1,l2,b0,b1,b2,vis = True):
	r = len(l0[0,:])
	s = len(l1[0,:])
	t = len(l2[0,:])
	b = np.concatenate([b0,b1,b2])
	loc = np.block([l0,l1,l2]) 

	#generate distance matrix
	d = np.zeros((r,s,t))
	for i in range(r):
		for j in range(s):
			for k in range(t):
				# avg_pt = np.array([(loc[0,i]+loc[0,j+r]+loc[0,k+r+t])/3,(loc[1,i]+loc[1,j+r]+loc[1,k+r+t])/3])
				# d[i,j,k]=dist.euclidean(loc[:,i],avg_pt)+dist.euclidean(loc[:,s+j],avg_pt)+dist.euclidean(loc[:,s+r+k],avg_pt)
				d[i,j,k]=dist.euclidean(l0[:,i],l1[:,j])+dist.euclidean(l1[:,j],l2[:,k])+dist.euclidean(l2[:,k],l0[:,i])

	A = buildA(r,s,t)

	#flatten the distance matrix into a cost vector
	c = np.ndarray.flatten(d)

	#solve the lp
	res = opt.linprog(c, A_eq=A, b_eq=b)
	T_flat= res.get('x')

	#reshape the transport matrix
	T = T_flat.reshape(r,s,t)

	#visualize results
	if vis == True:
		count = 0
		for i in range(r):
			for j in range(s):
				for k in range(t):
					if T[i,j,k] != 0:
						count = count+1
						plt.plot([l0[0,i],l1[0,j]], [l0[1,i],l1[1,j]], color='black', linewidth = T[i,j,k]*10, zorder=1)
						plt.plot([l0[0,i],l2[0,k]], [l0[1,i],l2[1,k]], color='black', linewidth = T[i,j,k]*10, zorder=1)
						plt.plot([l1[0,j],l2[0,k]], [l1[1,j],l2[1,k]], color='black', linewidth = T[i,j,k]*10, zorder=1)
		print(count)
		plt.scatter(l0[0,:], l0[1,:], color='red',s=b0*500, zorder=2)
		plt.scatter(l1[0,:], l1[1,:], color='green',s=b0*500, zorder=2)
		plt.scatter(l2[0,:], l2[1,:], color='blue',s=b0*500, zorder=2)
		plt.axis('equal')
		plt.show()

	return T

def run_example_centers(l0,l1,l2,b0,b1,b2,vis = True, f_w = True):
	r = len(l0[0,:])
	s = len(l1[0,:])
	t = len(l2[0,:])
	b = np.concatenate([b0,b1,b2])
	loc = np.block([l0,l1,l2]) 

	#generate distance matrix
	d = np.zeros((r,s,t))
	avg_pt = np.zeros((r,s,t,2))
	if f_w == True:
		for i in range(r):
			for j in range(s):
				for k in range(t):
					avg_pt[i,j,k,:] = geometric_median(np.array([list(l0[:,i]),list(l1[:,j]),list(l2[:,k])]))
					d[i,j,k]=dist.euclidean(l0[0,i],avg_pt[i,j,k])+dist.euclidean(l1[0,j],avg_pt[i,j,k])+dist.euclidean(l2[0,k],avg_pt[i,j,k])
					#d[i,j,k]=dist.euclidean(loc[:,i],loc[:,k+j])+dist.euclidean(loc[:,s+j],loc[:,s+r+k])+dist.euclidean(loc[:,s+r+k],loc[:,i])
	else:
		for i in range(r):
			for j in range(s):
				for k in range(t):
					avg_pt[i,j,k,:] = np.array([(l0[0,i]+l1[0,j]+l2[0,k])/3,(l0[1,i]+l1[1,j]+l2[1,k])/3])
					d[i,j,k]=dist.euclidean(l0[0,i],avg_pt[i,j,k])+dist.euclidean(l1[0,j],avg_pt[i,j,k])+dist.euclidean(l2[0,k],avg_pt[i,j,k])
					#d[i,j,k]=dist.euclidean(loc[:,i],loc[:,k+j])+dist.euclidean(loc[:,s+j],loc[:,s+r+k])+dist.euclidean(loc[:,s+r+k],loc[:,i])


	A = buildA(r,s,t)

	#flatten the distance matrix into a cost vector
	c = np.ndarray.flatten(d)

	#solve the lp
	res = opt.linprog(c, A_eq=A, b_eq=b)
	T_flat= res.get('x')

	#reshape the transport matrix
	T = T_flat.reshape(r,s,t)

	#visualize results
	if vis == True:
		count = 0
		for i in range(r):
			for j in range(s):
				for k in range(t):
					if T[i,j,k] != 0:
						count = count+1
						plt.plot([l0[0,i],avg_pt[i,j,k,0]], [l0[1,i],avg_pt[i,j,k,0]], color='red', linewidth = T[i,j,k]*10, zorder=1)
						plt.plot([l1[0,j],avg_pt[i,j,k,0]], [l1[1,j],avg_pt[i,j,k,0]], color='green', linewidth = T[i,j,k]*10, zorder=1)
						plt.plot([l2[0,k],avg_pt[i,j,k,0]], [l2[1,k],avg_pt[i,j,k,0]], color='blue', linewidth = T[i,j,k]*10, zorder=1)
		print(count)
		plt.scatter(l0[0,:], l0[1,:], color='red',s=b0*500, zorder=2)
		plt.scatter(l1[0,:], l1[1,:], color='green',s=b0*500, zorder=2)
		plt.scatter(l2[0,:], l2[1,:], color='blue',s=b0*500, zorder=2)
		plt.axis('equal')
		plt.show()

	return T

#THIS IS MESSED UP
# def geometric_median(X, eps=1e-5):
#     y = np.mean(X, 0)

#     while True:
#         D = cdist(X, [y])
#         nonzeros = (D != 0)[:, 0]

#         Dinv = 1 / D[nonzeros]
#         Dinvs = np.sum(Dinv)
#         W = Dinv / Dinvs
#         T = np.sum(W * X[nonzeros], 0)

#         num_zeros = len(X) - np.sum(nonzeros)
#         if num_zeros == 0:
#             y1 = T
#         elif num_zeros == len(X):
#             return y
#         else:
#             R = (T - y) * Dinvs
#             r = np.linalg.norm(R)
#             rinv = 0 if r == 0 else num_zeros/r
#             y1 = max(0, 1-rinv)*T + min(1, rinv)*y

#         if euclidean(y, y1) < eps:
#             return y1

#         y = y1


#generate equality constraints matrix
def buildA(r,s,t):
	A = np.zeros((r+s+t,r*s*t))
	for i in range(r):
		A[i,i*s*t:(i+1)*s*t] = 1

	for i in range(s):
		for j in range(r):
			A[i+r,j*s*t+i*t:j*s*t+t+i*t] = 1

	for i in range(t):
		for j in range(r*s*t):
			if Mod(j-i, t) == 0:
				A[i+r+s,j] = 1
	return A

T = run_example_centers(l0,l1,l2,b0,b1,b2,f_w = False)
#T = run_example_centers(l0,l1,l2,b0,b1,b2)
T = run_example_triangles(l0,l1,l2,b0,b1,b2)



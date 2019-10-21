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



#uncomment this for a simple example
# l0 = np.array([[0,0,0],[0,1,2]])
# l1 = np.array([[2,2],[.5,1.5]])
# b0 = np.array([1/3,1/3,1/3])
# b1 = np.array([1/2,1/2])

#sample two probability distributions same probalilities
# l0 = np.random.normal(loc = 0, size = (2,10))
# l1 = np.random.normal(loc = 0, size = (2,5))
# b0 = np.ones(10)/10
# b1 = np.ones(5)/5

# #sample 5 points from two probability distributions
l0 = np.random.normal(loc = 0, size = (2,20))
l1 = np.random.normal(loc = 0, size = (2,10))
b0 = np.random.random(20)
b0 = b0/sum(b0)
b1 = np.random.random(10)
b1 = b1/sum(b1)

#sample points circle
#FIX
# circle_x = 100
# circle_y = 100
# a = random.randint(0,100) * 2 * math.pi
# r = 1 * math.sqrt(random.randint(0,100))
# l0 = [[r * math.cos(a) + circle_x],[r * math.sin(a) + circle_y]]


#user parameters
#location of the support of distrtibution 0 and distribution 1.
# l0 = np.array([[0,1,0,1],[0,0,1,1]])
# l1 = np.array([[.25,.75,.5],[.25,.75,.5]])
# #probabilities for each point in the support of distriobution 0 and distribution 1
# b0 = np.array([1/4,1/4,1/4,1/4])
# b1 = np.array([1/3,1/3,1/3])

T,cost = run_example(l0,l1,b0,b1)



def run_example(l0,l1,b0,b1,vis = True):
	r = len(l0[0,:])
	s = len(l1[0,:])
	b = np.concatenate([b0,b1])
	loc = np.block([l0,l1]) 

	#generate distance matrix
	d = np.zeros((r,s))
	for i in range(r):
		for j in range(s):
			d[i,j]=dist.euclidean(loc[:,i],loc[:,j+r])

	A = buildA(r,s)

	#flatten the distance matrix into a cost vector
	c = np.ndarray.flatten(d)

	#solve the lp
	res = opt.linprog(c, A_eq=A, b_eq=b)
	t_flat= res.get('x')

	#reshape the transport matrix
	t = t_flat.reshape(r,s)

	#visualize results
	if vis == True:
		count = 0
		for i in range(r):
			for j in range(s):
				if t[i,j] != 0:
					count = count+1
					w = .1*t[i,j]
					plt.plot([l0[0,i],l1[0,j]], [l0[1,i],l1[1,j]], color='black', linewidth = t[i,j]*10, zorder=1)
		print(count)
		plt.scatter(loc[0,:r], loc[1,:r], color='green',s=b0*500, zorder=2)
		plt.scatter(loc[0,r:],loc[1,r:], color='red',s=b1*500, zorder=2)
		plt.axis('off')
		plt.show()

	return t, np.trace(d.T@t)



#generate equality constraints matrix
def buildA(r,s):
	A = np.zeros((r+s,r*s))
	for i in range(r):
		for j in range(s):
			A[i,j+s*i] = 1
	for i in range(s):
		for j in range(r*s):
			if Mod(j-i, s) == 0:
				A[i+r,j] = 1
	return A



import numpy as np

# Solving the flow
def flow(u,x,odestep,h,steps):
	'''
	  u: vector field
	  x: n-by-2 array of initial conditions where n is the grid size
	  odestep: numerical method for solving ode
	  h: step size for numerical method
	  steps: total number of steps to take

	  The algorithm returns Points which is a steps-by-n-by-2 array.
	  Points[i,j,k] is the value of the kth coordinate of the jth point
	  at the ith time step. 
	'''
	Points = np.empty((steps,len(x),2))
	Points[0,:,:] = x
	for i in range(1,steps):
		x = odestep(u,x,h)
		x = x - np.floor(x)
		Points[i,:,:] = x
	return Points

# Vector field
def u(x):
	return [x[1], -x[0]]

# Euler method
def euler(u,x,h):
	return x + h * np.array([u(xi) for xi in x])


x = np.mgrid[0:1:0.3, 0:1:0.3].reshape(2,-1).T
h = 0.01
steps = 5

points = flow(u,x,euler,h,steps)
print(points)
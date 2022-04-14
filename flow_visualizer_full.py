# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 20:57:44 2022

@author: lfsto
"""

#%matplotlib nbagg

import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import time
import os
import imageio

##### Solving the flow #####

def flow(u, x, odestep, h, steps):
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
	Points = np.empty((steps,*x.shape))
	Points[0] = x
	for i in range(1,steps):
		x = odestep(u,x,h)
		x = x - np.floor(x)
		Points[i] = x
	return Points


##### ODE solvers #####

# Euler
def euler(u,x,h):
	return x + h * u(*x)

# Runge-Kutta 4
def rk4(u,x,h):
	k1 = u(*x)
	k2 = u(*(x + (h/2) * k1))
	k3 = u(*(x + (h/2) * k2))
	k4 = u(*(x + h * k3))
	return x + (h/6) * (k1 + 2.*(k2 + k3) + k4)


##### Vector field #####
'''
# At a point
def u_point(x):
	return np.array([x[1], -x[0]])

# Over the whole space
def u(x):
	return np.array([u_point(xi) for xi in x])
'''

##### Initial data #####

def initial(x):
	initial_vals = np.ones_like(x)
	for i in np.ndindex(x.shape):
		if x[i] > 0.5:
			initial_vals[i] = 0.
	return initial_vals

def sin(x):
    z = 2*(x%1)-1
    return sum([(-1)**(n//2) * np.pi**n * z**n / np.math.factorial(n) for n in range(1,20,2)])

def cos(x):
    z = 2*(x%1)-1
    return sum([(-1)**(n//2+1) * np.pi**n * z**n / np.math.factorial(n) for n in range(0,20,2)])


def make_field(n):
    const = np.random.rand(2)*2-1
    coeff = np.random.rand(n,n,4)*2-1
    
    def velocity(x, y):
        total = np.array([np.full_like(x,const[0]),np.full_like(x,const[1])])
        
        cos_x_vals = {}
        sin_x_vals = {}
        cos_y_vals = {}
        sin_y_vals = {}
        
        for i in range(n):
            cos_x_vals[i] = cos(i*x)
            sin_x_vals[i] = sin(i*x)
        for j in range(n):
            cos_y_vals[j] = cos(j*y)
            sin_y_vals[j] = sin(j*y)
        
        for i,j in zip(range(n),range(n)):
            size_opp = (coeff[i,j,0] * (sin_x_vals[i] * sin_y_vals[j] - cos_x_vals[i]*cos_y_vals[j])
                      + coeff[i,j,1] * (sin_x_vals[i] * cos_y_vals[j] + cos_x_vals[i]*sin_y_vals[j])
                      ) / (1+i**2+j**2)
            size_same= (coeff[i,j,2] * (sin_x_vals[i] * sin_y_vals[j] + cos_x_vals[i]*cos_y_vals[j])
                      + coeff[i,j,3] * (sin_x_vals[i] * cos_y_vals[j] - cos_x_vals[i]*sin_y_vals[j])
                      ) / (1+i**2+j**2)
            total += [j*(size_same - size_opp), i*(size_same+size_opp)]
        
        return total
    
    return velocity




def run_model(grid_resolution, frequency_range, num_time_steps):
    global u, points,grid,initial_vals,h,steps,n,freq
    n = grid_resolution
    freq = frequency_range
    grid = np.array(np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n)))
    initial_vals = initial(grid[0])
    h = 0.01
    steps = num_time_steps
    
    
    u = make_field(frequency_range)
    points = flow(u, grid, rk4, h, steps)


### Animation ###
'''
fig, ax = plt.subplots(1, 1)

colors = np.where(initial_vals.flatten()==1,'yellow','black')
scatter = ax.scatter(points[0,0].flatten(), points[0, 1].flatten(), s=3, c=colors)



def update(i):
   scatter.set_offsets(points[i,0].flatten(),points[i,1].flatten())
   return scatter,

anim = FuncAnimation(fig, update, frames=steps, interval=10, repeat=True)

#anim.save('Simulation3.mp4', fps=30)
plt.show()
'''


def gif():    #(x_vals, y_vals):
    filenames = []
    save_as = f"REU_Data_{time.strftime('%Y-%m-%d-%H.%M.%S')}.gif"
    
    fig, ax = plt.subplots(1, 1)
    
    colors = np.where(initial_vals.flatten()==1,'yellow','black')
    scatter = ax.scatter(points[0,0].flatten(), points[0, 1].flatten(), s=3, c=colors)
    ax.set_title(f"resolution {n}, u frequencies up to {freq}")
    
    for i in range(steps):
        data = list(map(list,zip(points[i,0].flatten(),points[i,1].flatten())))
        
        scatter.set_offsets(data) #np.c_[data[::2], data[1::2]])
        filename = f"temp_image_{i}.png"
        filenames.append(filename)
        fig.savefig(filename)
        #fig.close()
    # build gif
    with imageio.get_writer(save_as, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)
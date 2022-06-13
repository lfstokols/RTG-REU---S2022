#----#----#----#----#----#----#----#----#----#----#----#----#
#----#
#----# Exectute this file, 
#----# Then run run_model() with parameters for 
#----#   the spatial resolution, the time steps, and frequency cutoff.
#----# Then run gif(), no parameters
#----#
#----# Note that gif() will save files to your computer! 
#----# GIF() WILL SAVE FILES TO YOUR COMPUTER!!!!!
#----# GIF() WILL SAVE HUNDREDS OF FILES TO YOUR COMPUTER!!!!!!
#----# It will save hundreds of files, 
#----# then delete them all (assuming it fully executes), then save 
#----# one final .gif
#----#
#----#----#----#----#----#----#----#----#----#----#----#----#


#%matplotlib nbagg

import numpy as np
from scipy.fft import fft
from scipy.interpolate import griddata
import matplotlib.pyplot as plt  
#import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation
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
def make_field(n,intial_value,alpha = 2.25, sigma = 1.0, seed = 0):
    '''
    n: Number of Fourier modes used to generate the field
    alpha: power of Xi function, Default set to 2.25
    sigma: Scales the standard deviation (sigma)of the normally randomly distributed coefficents [0.1,...,2.0]
           Default set to 1.0
    seed: Seed of random number sequence used in the simmulation. Default set to 0.    
    '''
    from numpy import linalg as LA
    n += 1
    x = intial_value
    np.random.seed(seed)   
    total = np.array([np.full_like(x,0),np.full_like(x,0)])
    coeff = np.zeros((n,n,4))
    scale = sigma*np.ones(4)

    coeff[:,:,0] = scale[0]*np.random.normal(np.zeros((n,n)))
    coeff[:,:,1] = scale[1]*np.random.normal(np.zeros((n,n)))
    coeff[:,:,2] = scale[2]*np.random.normal(np.zeros((n,n)))
    coeff[:,:,3] = scale[3]*np.random.normal(np.zeros((n,n)))

    coeff[:,:,0] = 2*np.pi*coeff[:,:,0]/LA.norm(coeff[:,:,0], 'fro')
    coeff[:,:,1] = 2*np.pi*coeff[:,:,1]/LA.norm(coeff[:,:,1], 'fro')
    coeff[:,:,2] = 2*np.pi*coeff[:,:,2]/LA.norm(coeff[:,:,2], 'fro')
    coeff[:,:,3] = 2*np.pi*coeff[:,:,3]/LA.norm(coeff[:,:,3], 'fro')
    
    def velocity(x, y):
        total = np.array([np.full_like(x, 0),np.full_like(x, 0)])
        cos_x_vals = {}
        sin_x_vals = {}
        cos_y_vals = {}
        sin_y_vals = {}
        Xi = np.ones((n,n))
        
        for i in range(n):
            cos_x_vals[i] = np.cos(2*np.pi*i*x)
            sin_x_vals[i] = np.sin(2*np.pi*i*x)
            for j in range(n):
                Xi[i,j] = (1.0/(1+i*i+j*j))**(alpha/2.0)
                cos_y_vals[j] = np.cos(2*np.pi*j*y)
                sin_y_vals[j] = np.sin(2*np.pi*j*y)

        for i,j in zip(range(n),range(n)):
            u  =  coeff[i,j,0]*i*Xi[i,j]*cos_x_vals[i]*sin_y_vals[j]
            u +=  coeff[i,j,1]*i*Xi[i,j]*cos_x_vals[i]*cos_y_vals[j]
            u += -coeff[i,j,2]*i*Xi[i,j]*sin_x_vals[i]*sin_y_vals[j]
            u += -coeff[i,j,3]*i*Xi[i,j]*sin_x_vals[i]*cos_y_vals[j]
            v  =  coeff[i,j,0]*j*Xi[i,j]*sin_x_vals[i]*cos_y_vals[j]
            v += -coeff[i,j,1]*j*Xi[i,j]*sin_x_vals[i]*sin_y_vals[j]
            v +=  coeff[i,j,2]*j*Xi[i,j]*cos_x_vals[i]*cos_y_vals[j]
            v += -coeff[i,j,3]*j*Xi[i,j]*cos_x_vals[i]*sin_y_vals[j]
            total += [-v,u]
    
        return total  # advected field

    return velocity

##### Initial data #####

def initial(x):
    '''
    Input: nxn array of x-values

    Output: nxn array equalling 1/2 for points whose x-value is <0.5, and -1/2 else
    '''
    assert np.amax(x) <= 1 and np.amin(x) >= 0, "position values aren't between 0 and 1"
    initial_vals = np.where(x<0.5, np.full_like(x,0.5), np.full_like(x,-0.5))
    return initial_vals



##### Create full flow data #####

def run_model(grid_resolution, frequency_range, num_time_steps,alpha = 2.25, sigma = 1.0, seed = 0):
    '''
    Parameters:
    grid_resolution: Specifies the smoothness (high value) or corseness of the grid (low value).
    frequency_range: Number of Fourier modes used to generate the field
    num_time_steps: Number of time steps taken
    alpha: power of Xi function, Default set to 2.25
    sigma: Scales the standard deviation of the normally randomly distributed coefficents [0.1,...,2.0]
           Default set to 1.0
    seed: Seed of random number sequence used in the simmulation. Default set to 0. Good values to try 5, 10, 20     
    '''
    global u, points,grid,initial_vals,h,steps,n,freq, sig, s
    n = grid_resolution
    freq = frequency_range
    sig = sigma
    s = seed
    grid = np.array(np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n)))
    initial_vals = initial(grid[0])
    h = 0.02
    steps = num_time_steps
    u = make_field(frequency_range,initial_vals,alpha,sigma,seed)
    points = flow(u, grid, rk4, h, steps)

    return (points, initial_vals)




def make_gif(points, initial_vals, folder_path = ""):    #(x_vals, y_vals):
    filenames = []
    save_as = f"{folder_path}Flow_vis_{time.strftime('%Y-%m-%d-%H.%M.%S')}.gif"
    
    fig, ax = plt.subplots(1, 1)
    
    midpoint = (np.amax(initial_vals)+np.amin(initial_vals))/2
    colors = np.where(initial_vals.flatten()>midpoint,'red','blue')
    scatter = ax.scatter(points[0,0].flatten(), points[0, 1].flatten(), s=3, c=colors)
    ax.set_title(f"resolution {n}, u frequencies up to {freq} \n t={0:4}")
    
    for i in range(steps):
        data = list(map(list,zip(points[i,0].flatten(),points[i,1].flatten())))
        
        scatter.set_offsets(data) #np.c_[data[::2], data[1::2]])
        ax.set_title((f"resolution {n}, u frequencies up to {freq} \n t={i:4}"))
        filename = f"temp_image_{i}.png"
        filenames.append(filename)
        fig.savefig(filename)
    plt.close(fig)
    # build gif
    with imageio.get_writer(save_as, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)



'''
def initial(x):
    ini = np.ones((res, res))
       #ini1, ini2 = np.meshgrid(np.linspace(1,1,512), np.linspace(1,1,512))
    for i in range(res):
        if i >= res/2:
            ini[i] = 0.
                #ini2[i] = 0.
    return ini
'''

##### Measure H^-1 norm #####

def h1(data, initial_vals, t, res):
    '''
    Output: H^-1 norm at one timestep

    Inputs:
    data: points data from model run, shape (T,2,N,N) representing x and y values of initial NxN points at times up to T
    initial_vals: NxN array, represents initial value of function (.5 or -.5) at point with corresponding x,y coordinates
    t: the time step in data where H^-1 norm will be evaluated
    res: the resolution for the grid that data will be interpolated onto
        res should be a power of 2 (e.g. 1024) for best performance
        In principle res should equal N (size of data array) but not necessary
    '''
    #Check that data is of shape (T,2,N,N) where t less than T=total number of timesteps
    assert len(data.shape) == 4 and t < data.shape[0] and data.shape[1] == 2 and data.shape[2] == data.shape[3] and initial_vals.shape == data.shape[2:], "Data is miss-formatted"
    # define grid points
    data_x = data[t][0].flatten()
    data_y = data[t][1].flatten()
    # z-axis refers to the function value at point, either 0.5 or -0.5
    data_z = initial_vals.flatten()
    # interpolate the data to lie on an evenly-spaced resxres grid
    x, y = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
    dn = griddata((data_x, data_y), (data_z), (x, y), 'nearest').reshape(res,res)
    # take fft of the points, normalize by res**2 for 1x1 torus
    data_ff = np.fft.fft2(dn-np.mean(dn)) / res**2
    # The FFT value at (j,k) only corresponds to the (j,k) frequency if j,k < res/2. 
    # But e.g. the res-1 value corresponds to frequency -1, not res-1
    # Here we create an array representing j^2+k^2
    lin = np.mod(np.linspace(0,res-1,res) + res/2, res) - res/2
    J,K = np.meshgrid(lin, lin)
    denom = J**2 + K**2
    # At [0,0] we should have 0/0 = 0, set denom to 1 to avoid NaN errors
    denom[0,0] = 1
    # now we can compute L^2 norm of data_ff / sqrt(denom)
    return np.sqrt( np.sum(np.abs(data_ff)**2 / denom) ) 

##### Plot H^-1 norm as function of t #####

def measure_mixing(data, initial_vals, res, show=1):
    '''
    Output: Graphs the H^-1 norm as function of time, both returning array and plotting as graph
    Input:
    data: points data from model run, shape (T,2,N,N) representing x and y values of initial NxN points at times up to T
    initial_vals: NxN array, represents initial value of function (.5 or -.5) at point with corresponding x,y coordinates
    res: the resolution for the grid that data will be interpolated onto
        res should be a power of 2 (e.g. 1024) for best performance
        In principle res should equal N (size of data array) but not necessary
    show: by default, shows graph. If show != 1, suppress graph
    '''
    # Check that data is of shape (T,2,N,N)
    # Note that N (resolution of initial data) need not equal res (resolution of interpolated data)
    assert len(data.shape) == 4 and data.shape[1] == 2 and data.shape[2] == data.shape[3] and initial_vals.shape == data.shape[2:], "Data is miss-formatted"
    num_time_steps = data.shape[0]
    
    # run h1 in for-loop
    Scale = np.empty(num_time_steps)
    for t in range(num_time_steps):
        Scale[t] = h1(data, initial_vals, t, res)
    
    # graph the data
    plt.plot(range(num_time_steps), Scale)
    if show==1:
        plt.show()

    # return data in case you want to analyze it
    return Scale
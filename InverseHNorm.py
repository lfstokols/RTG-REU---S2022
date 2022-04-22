import numpy as np
from scipy.fft import fft
from scipy.interpolate import griddata
import cmath as cm

# define grid points
data = np.load('REU_Model_run_200x512x512.npy')
data1 = data[199][0]
data2 = data[199][1]
def initial(x):
        ini = np.ones((512, 512))
        #ini1, ini2 = np.meshgrid(np.linspace(1,1,512), np.linspace(1,1,512))
        for i in range(512):
            if x[0,i] > 0.5:
                ini[i] = 0.
                #ini2[i] = 0.
        return ini
def repeat(x):
    vals = []
    for i in range(512):
        vals.append(initial(x))
    return vals

datanew = initial(data1).flatten()
x, y = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
dn = griddata((data1.flatten(), data2.flatten()), (datanew), (x, y), 'nearest').reshape(512,512)

# take fft of the points
data_ff = fft(dn)
xn = np.zeros((512,512))
# divide them by sqrt of j^2 + k^2 (where j and k are the indices of the array of grid points)
for j in range(512):
    for k in range(512):
        xn[j,k] = data_ff[j, k] / cm.sqrt((j**2) + (k**2))

# sum all those new points, raise them to the power of 2, and take sqrt again
xn = np.abs(xn)
xn1 = xn**2
x2 = sum(sum(xn1))
xf = np.sqrt(x2)

print(xf)
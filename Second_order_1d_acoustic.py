import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import gridspec
from drawnow import drawnow

        
def makeFig():

    # plt.plot(hnew) 
    plt.scatter(x,h) 


def rd(x): # Eucldidean distance    

    x1 = x.reshape(-1,1)
    x2 = x1.T
    rd2 = (x1-x2)**2
    rd1 = (x1-x2)
    rabs = np.sqrt((x1 - x2)**2)
    rabs_inv = np.linalg.inv(rabs)

    return rabs,rd1,rd2,rabs_inv


def RBF_DM(x):
    
    eps = 1.1
    radius,rd1,rd2,radius_inv = rd(x)
    A = np.exp(-eps**2*rd2)
    A_inv = np.linalg.inv(A)
    B = -rd1
    B = 2*(eps**2)*B*np.exp(-(eps**2)*rd2)
    D = np.matmul(B,A_inv)
    
    return D


def FD_der2(u,dx):

    fd2x = (u[2:] - 2*u[1:-1] + u[:-2])/dx**2

    return fd2x

def compute_rbf_and_D(x,TYPE='GAUSS_RBF'):

    if TYPE == 'GAUSS_RBF':

        eps = 0.85
        radius,rd1,rd2,radius_inv = rd(x)
        rbf = np.exp(-eps**2*rd2)
        rbf2 = np.exp(-2*eps**2*rd2)
        A_inv = np.linalg.inv(rbf)
        B = -2*(eps**2)*rd1*rbf
        # B2 = 4*eps**4*rd2*rbf2 
        D = np.matmul(B,A_inv)

    elif TYPE == 'POLY_HARM':
    
        k = 5
        radius,rd1,rd2 = rd(x)

        if (k % 2) != 0:
            rbf = radius**k
        else:
            rbf = radius**(k-1) * np.log(radius**radius)

        A_inv = np.linalg.inv(rbf)
        B = k*radius*rbf**(k-2)
        D = np.matmul(B,A_inv)

    return rbf,D


def compute_rbf(radius,TYPE='GAUSS_RBF'):

    if TYPE == 'GAUSS_RBF':

        eps = 1
        rbf = np.exp(-(eps*radius)**2)

    elif TYPE == 'POLY_HARM':
    
        k = 5 
        if (k % 2) != 0:
            rbf = radius**k
            # rbf = k*radius**(k-1)
        else:
            rbf = radius**(k-1) * np.log(radius**radius)

    return rbf

# Definition of modelling parameters
# ----------------------------------
c = 1500 # speed of sound in water (m/s)
wavelenght = 100 # meters

nwave = 60 # number of wavelenght to propagate the simulation
npts_per_wave = 50 # number of points per wavelenght

nx = int(nwave*npts_per_wave) # number of grid points in x-direction
xmax = nwave*wavelenght # total length
dx = xmax/nx # spatial discretization

x = np.linspace(0,xmax,nx)

COURANT = 1

dt = COURANT*dx/c 

# total time to run two times the total lenght
tmax = 2*xmax/c

nt = int(tmax/dt) # maximum number of time steps

# acquisition geometry
xr = int(2*xmax/3) # receiver position (m)
xsrc = int(xmax/2) # source position (m)  
    
ir = np.argmin(abs(x-xr))   # receiver location in grid in x-direction    
isrc = np.argmin(abs(x-xsrc))  # source location in grid in x-direction

# Source time function 
f0   = c/wavelenght # dominant frequency of the source (Hz)
t0   = 4 / f0 # source time shift (s)
time = np.linspace(0,tmax,nt)

# 1st derivative of a Gaussian
src  = -2 * (time - t0) * (f0 ** 2) * (np.exp(- (f0 ** 2) * (time - t0) ** 2))

PLOT_SRC = True
if PLOT_SRC:
    plt.figure(2)
    plt.plot(time,src)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title("Source Time Function")
    plt.grid()
    plt.show()

# Initialize empty velocty/stress arrays 
# --------------------------------
h,hold,hnew,fd2x = np.zeros((4,nx)) 

 # Initialize empty seismogram
 # ---------------------------
seis = np.zeros(nt) 

rbf,D = compute_rbf_and_D(x,TYPE='GAUSS_RBF')

SAMPLING = 50

# Time stepping using method of lines approach
# -----------------------------
for it in range(nt):
    
    print('Time level: ',it)

    h[isrc] = dt**2*src[it]/dx
    
    # finite difference derivative
    #fd2x[1:-1] = FD_der2(h,dx)

    # radial basis function second derivaitve
    fd2x = D.dot(D.dot(h))

    # finite difference extrapolation
    hnew = 2*h - hold + dt**2*c**2*fd2x

    # Remap Time Levels
    # -----------------
    hold[:],h[:] = h[:],hnew[:]
    
    # Output Seismogram
    # -----------------
    seis[it] = hnew[ir]
    
    # plot resutls
    if it % SAMPLING == 0:
        drawnow(makeFig)


plt.figure(3)
plt.plot(time,seis)
plt.xlabel('time in seconds')
plt.ylabel('Amplitude')
plt.title('Seismometer reading of Finite Difference method')
plt.show()

plt.figure(4)
plt.plot(x,h)
plt.scatter(xsrc,0,c='r',marker = '*',s = 20*16, label = 'source')
plt.scatter(xr,0,c='g',marker = 'v',s=20*16, label = 'receiver')
plt.xlabel('x')
plt.ylabel('particle displacement')
plt.title('RBF Method')
plt.legend()
plt.savefig('plot_x_h_rbf.png')
plt.show()


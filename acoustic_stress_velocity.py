import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from matplotlib import gridspec
from drawnow import drawnow

        
def makeFig():

    plt.scatter(x,vel)  
    # plt.scatter(x,vel)
    # plt.plot(seis)


def rd(x): # Eucldidean distance    

    x1 = x.reshape(-1,1)
    x2 = x1.T
    rd2 = (x1 - x2)**2
    rd1 = (x1-x2)
    rabs = np.sqrt((x1 - x2)**2)

    return rabs,rd1,rd2


# def euclidean_dist(x,xk):

#     dist = np.sqrt((x.reshape(-1,1) - xk.reshape(1,-1))**2)

#     return dist


def RBF_DM(x):
    
    eps = 1.1
    radius,rd1,rd2 = rd(x)
    A = np.exp(-eps**2*rd2)
    A_inv = np.linalg.inv(A)
    B = -rd1
    B = 2*(eps**2)*B*np.exp(-(eps**2)*rd2)
    D = np.matmul(B,A_inv)
    
    return D


def FD_der_back(fdx,u,dx):

    fdx[1:] = (u[:-1]-u[1:])/dx

    return fdx

def FD_der_fow(fdx,u,dx):

    fdx[1:] = (u[1:]-u[:-1])/dx

    return fdx

def compute_rbf_and_D(x,TYPE='GAUSS_RBF'):

    if TYPE == 'GAUSS_RBF':

        eps = 0.75 # 1.1
        radius,rd1,rd2 = rd(x)
        rbf = np.exp(-eps**2*rd2)
        A_inv = np.linalg.inv(rbf)
        B = -2*(eps**2)*rd1*rbf
        # D = B # following me this is the derivative
        D = np.matmul(B,A_inv)

    elif TYPE == 'POLY_HARM':
    
        k = 3
        radius,rd1,rd2 = rd(x)

        if (k % 2) != 0:
            rbf = radius**k
        else:
            rbf = radius**(k-1) * np.log(radius**radius)

        A_inv = np.linalg.inv(rbf)
        B = rd1*k*radius**(k-2)
        D = np.matmul(B,A_inv)

    return rbf,D


def compute_rbf(radius,TYPE='GAUSS_RBF'):

    if TYPE == 'GAUSS_RBF':

        eps = 3
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
c = 900 # speed of sound in water (m/s)
wavelenght = 100 # meters

rho = 1800 # kg/m3
lame = c**2*rho # lame parameter

nwave = 50 # number of wavelenght to propagate the simulation
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
vel,sigma,fvdx,fsdx = np.zeros((4,nx)) 

 # Initialize empty seismogram
 # ---------------------------
seis = np.zeros(nt) 

# D = RBF_DM(x)
rbf,D = compute_rbf_and_D(x,TYPE='GAUSS_RBF')

SAMPLING = 50

# Time stepping using method of lines approach
# -----------------------------
for it in range(nt):
    
    print('Time level: ',it)
    
    fvdx = D.dot(vel)
    sigma += lame*dt*fvdx
    
    fsdx = D.dot(sigma)
    vel += dt/rho * fsdx

    vel[isrc] += dt*src[it]/dx
    
    # velocity extrapolation
    # fsdx[:-1] = (sigma[1:] - sigma[:-1])/dx
    #fsdx = D.dot(sigma)
    #vel += dt/rho * fsdx

    # vel[isrc] = dt*src[it]/dx
    
    # stress extrapolation
    # fvdx[1:] = (vel[1:] - vel[:-1])/dx
    #fvdx = D.dot(vel)
    #sigma += lame*dt*fvdx
    
    # 4th order (not working)
    # d1 = dt/rho*D.dot(sigma)
    # d2 = dt/rho*D.dot(sigma + 0.5*d1)
    # d3 = dt/rho*D.dot(sigma + 0.5*d2)
    # d4 = dt/rho*D.dot(sigma + d3)
    # vel += 1/6*(d1 + 2*d2 + 2*d3 + d4)
    
    # d1 = lame*dt*D.dot(vel)
    # d2 = lame*dt*D.dot(vel + 0.5*d1)
    # d3 = lame*dt*D.dot(vel + 0.5*d2)
    # d4 = lame*dt*D.dot(vel + d3)
    # sigma += 1/6*(d1 + 2*d2 + 2*d3 + d4)
        
    # Output Seismogram
    # -----------------
    seis[it] = vel[ir]
    
    # plot resutls
    if it % SAMPLING == 0:
        drawnow(makeFig)

plt.figure(3)
plt.plot(time,seis)
plt.xlabel('time in seconds')
plt.ylabel('Amplitude')
plt.title('Seismometer reading of RBF method')
plt.savefig('vel_seis_plot')
plt.show()

plt.figure(4)
plt.plot(x,vel)
plt.scatter(xsrc,0, c='red',marker ='*',s = 20*16, label = 'source')
plt.scatter(xr,0, c='green',marker = 'v',s= 20*16, label = 'receiver')
plt.xlabel('x')
plt.ylabel('particle Velocity')
plt.title('RBF method')
plt.savefig('vel_plot')
plt.legend()
plt.show()

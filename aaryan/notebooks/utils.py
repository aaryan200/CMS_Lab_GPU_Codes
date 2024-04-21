import cunumeric as cun
import numpy as np
import pyfftw
from scipy.fft import fft2 as scipy_fft2
from scipy.fft import ifft2 as scipy_ifft2
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from legate.timing import time

def np_spinodal(arr_size = 100, num_iter = 10,
                 steps_per_iter = 100,
                verbose = True):
    #size of the box
    Nx = arr_size
    Ny = arr_size
    #Initial composition 
    c = 0.5*np.ones([Nx,Ny])
    #Noise - seed (mimic thermal fluctuations) 
    np.random.seed(1024)
    random_num = np.random.normal(0,0.01,(Nx,Ny))

    c = c - random_num

    if verbose:
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    dx = 1.0
    dy = 1.0
    dt = 0.5
    cnew = c
    #delkx is grid spacing along kx in Fourier space
    #delky is grid spacing along ky in Fourier space

    delkx = 2*np.pi/(Nx*dx)
    delky = 2*np.pi/(Ny*dy)

    #A is the prefactor of free energy f = A (c**2) * (1-c)**2
    #A is inversely proportional to temperature

    A = 1 

    #M is the scaled constant mobility where diffusivity D = M (d^2f/dc^2) 
    M = 1

    # kappa is the scaled gradient energy coefficient (interfacial energy)
    kappa = 1

    # Start time
    start_time = time("s")

    i_indices = np.arange(Nx) # Numbers from [0 to Nx-1]
    j_indices = np.arange(Ny) # Numbers from [0 to Ny-1]

    kx = np.where(i_indices <= Nx/2, i_indices * delkx, (i_indices - Nx) * delkx)
    # If i <= Nx/2, kx = i*delkx, else kx = (i-Nx)*delkx
    ky = np.where(j_indices <= Ny/2, j_indices * delky, (j_indices - Ny) * delky)
    # If j <= Ny/2, ky = j*delky, else ky = (j-Ny)*delky

    # Calculate k2 and k4
    # kx[:, np.newaxis] will reshape it to (Nx, 1)
    # ky[np.newaxis, :] will reshape it to (1, Ny)
    # Their squared addition will result in a (Nx, Ny) matrix
    # k2[first row] = [kx[0]**2 + ky[0]**2, kx[0]**2 + ky[1]**2, ..., kx[0]**2 + ky[Ny-1]**2]
    k2 = kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2
    k4 = k2**2

    # Outer iterations
    for m in range(num_iter):
        # Inner iterations
        for n in range(steps_per_iter):
            
            #g stores the first derivative of free energy
            #df/dc = 2*A*c*(1-c)*(1-2c)
            mult = np.multiply(1-cnew,1-2*cnew)
            g = 2*A*np.multiply(cnew,mult)
            ghat = np.fft.fft2(g)
            chat = np.fft.fft2(cnew)
            
            chat = (chat - M*dt*k2*ghat)/(1+2*M*kappa*k4*dt)

            cnew = np.fft.ifft2(chat).real
            c = cnew

        if verbose: print(m)

    end_time = time("s")

    if verbose:
        clear_output(wait=True)
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    if verbose:
        print("Execution Time:", execution_time, "seconds")
    return execution_time

def sp_spinodal(arr_size = 100, num_iter = 10,
                 steps_per_iter = 100,
                verbose = True):
    #size of the box
    Nx = arr_size
    Ny = arr_size
    #Initial composition 
    c = 0.5*np.ones([Nx,Ny])
    #Noise - seed (mimic thermal fluctuations) 
    np.random.seed(1024)
    random_num = np.random.normal(0,0.01,(Nx,Ny))

    c = c - random_num

    if verbose:
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    dx = 1.0
    dy = 1.0
    dt = 0.5
    cnew = c
    #delkx is grid spacing along kx in Fourier space
    #delky is grid spacing along ky in Fourier space

    delkx = 2*np.pi/(Nx*dx)
    delky = 2*np.pi/(Ny*dy)

    #A is the prefactor of free energy f = A (c**2) * (1-c)**2
    #A is inversely proportional to temperature

    A = 1 

    #M is the scaled constant mobility where diffusivity D = M (d^2f/dc^2) 
    M = 1

    # kappa is the scaled gradient energy coefficient (interfacial energy)
    kappa = 1

    # Start time
    start_time = time("s")

    i_indices = np.arange(Nx) # Numbers from [0 to Nx-1]
    j_indices = np.arange(Ny) # Numbers from [0 to Ny-1]

    kx = np.where(i_indices <= Nx/2, i_indices * delkx, (i_indices - Nx) * delkx)
    # If i <= Nx/2, kx = i*delkx, else kx = (i-Nx)*delkx
    ky = np.where(j_indices <= Ny/2, j_indices * delky, (j_indices - Ny) * delky)
    # If j <= Ny/2, ky = j*delky, else ky = (j-Ny)*delky

    # Calculate k2 and k4
    # kx[:, np.newaxis] will reshape it to (Nx, 1)
    # ky[np.newaxis, :] will reshape it to (1, Ny)
    # Their squared addition will result in a (Nx, Ny) matrix
    # k2[first row] = [kx[0]**2 + ky[0]**2, kx[0]**2 + ky[1]**2, ..., kx[0]**2 + ky[Ny-1]**2]
    k2 = kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2
    k4 = k2**2

    # Outer iterations
    for m in range(num_iter):
        # Inner iterations
        for n in range(steps_per_iter):
            
            #g stores the first derivative of free energy
            #df/dc = 2*A*c*(1-c)*(1-2c)
            mult = np.multiply(1-cnew,1-2*cnew)
            g = 2*A*np.multiply(cnew,mult)
            ghat = scipy_fft2(g)
            chat = scipy_fft2(cnew)
            
            chat = (chat - M*dt*k2*ghat)/(1+2*M*kappa*k4*dt)

            cnew = scipy_ifft2(chat).real
            c = cnew

        if verbose: print(m)

    end_time = time("s")

    if verbose:
        clear_output(wait=True)
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    if verbose:
        print("Execution Time:", execution_time, "seconds")
    return execution_time

def pyfftw_spinodal(arr_size = 100, num_iter = 10,
                 steps_per_iter = 100,
                verbose = True):
    #size of the box
    Nx = arr_size
    Ny = arr_size
    #Initial composition 
    c = 0.5*np.ones([Nx,Ny])
    #Noise - seed (mimic thermal fluctuations) 
    np.random.seed(1024)
    random_num = np.random.normal(0,0.01,(Nx,Ny))

    c = c - random_num

    if verbose:
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    dx = 1.0
    dy = 1.0
    dt = 0.5
    cnew = c
    #delkx is grid spacing along kx in Fourier space
    #delky is grid spacing along ky in Fourier space

    delkx = 2*np.pi/(Nx*dx)
    delky = 2*np.pi/(Ny*dy)

    #A is the prefactor of free energy f = A (c**2) * (1-c)**2
    #A is inversely proportional to temperature

    A = 1 

    #M is the scaled constant mobility where diffusivity D = M (d^2f/dc^2) 
    M = 1

    # kappa is the scaled gradient energy coefficient (interfacial energy)
    kappa = 1

    # Start time
    start_time = time("s")

    i_indices = np.arange(Nx) # Numbers from [0 to Nx-1]
    j_indices = np.arange(Ny) # Numbers from [0 to Ny-1]

    kx = np.where(i_indices <= Nx/2, i_indices * delkx, (i_indices - Nx) * delkx)
    # If i <= Nx/2, kx = i*delkx, else kx = (i-Nx)*delkx
    ky = np.where(j_indices <= Ny/2, j_indices * delky, (j_indices - Ny) * delky)
    # If j <= Ny/2, ky = j*delky, else ky = (j-Ny)*delky

    # Calculate k2 and k4
    # kx[:, np.newaxis] will reshape it to (Nx, 1)
    # ky[np.newaxis, :] will reshape it to (1, Ny)
    # Their squared addition will result in a (Nx, Ny) matrix
    # k2[first row] = [kx[0]**2 + ky[0]**2, kx[0]**2 + ky[1]**2, ..., kx[0]**2 + ky[Ny-1]**2]
    k2 = kx[:, np.newaxis]**2 + ky[np.newaxis, :]**2
    k4 = k2**2

    # Outer iterations
    for m in range(num_iter):
        # Inner iterations
        for n in range(steps_per_iter):
            
            #g stores the first derivative of free energy
            #df/dc = 2*A*c*(1-c)*(1-2c)
            mult = np.multiply(1-cnew,1-2*cnew)
            g = 2*A*np.multiply(cnew,mult)
            ghat = pyfftw.interfaces.numpy_fft.fft2(g)
            chat = pyfftw.interfaces.numpy_fft.fft2(cnew)
            
            chat = (chat - M*dt*k2*ghat)/(1+2*M*kappa*k4*dt)

            cnew = pyfftw.interfaces.numpy_fft.ifft2(chat).real
            c = cnew

        if verbose: print(m)

    end_time = time("s")

    if verbose:
        clear_output(wait=True)
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    if verbose:
        print("Execution Time:", execution_time, "seconds")
    return execution_time

def cun_spinodal(arr_size = 100, num_iter = 10,
                 steps_per_iter = 100,
                 verbose = True):
    #size of the box
    Nx = arr_size
    Ny = arr_size
    #Initial composition 
    c = 0.5*cun.ones([Nx,Ny])
    #Noise - seed (mimic thermal fluctuations) 
    cun.random.seed(1024)
    random_num = cun.random.normal(0,0.01,(Nx,Ny))

    c = c - random_num

    if verbose:
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    dx = 1.0
    dy = 1.0
    dt = 0.5
    cnew = c
    #delkx is grid spacing along kx in Fourier space
    #delky is grid spacing along ky in Fourier space

    delkx = 2*cun.pi/(Nx*dx)
    delky = 2*cun.pi/(Ny*dy)

    #A is the prefactor of free energy f = A (c**2) * (1-c)**2
    #A is inversely proportional to temperature

    A = 1 

    #M is the scaled constant mobility where diffusivity D = M (d^2f/dc^2) 
    M = 1

    # kappa is the scaled gradient energy coefficient (interfacial energy)
    kappa = 1

    # Start time
    start_time = time("s")

    i_indices = cun.arange(Nx) # Numbers from [0 to Nx-1]
    j_indices = cun.arange(Ny) # Numbers from [0 to Ny-1]

    kx = cun.where(i_indices <= Nx/2, i_indices * delkx, (i_indices - Nx) * delkx)
    # If i <= Nx/2, kx = i*delkx, else kx = (i-Nx)*delkx
    ky = cun.where(j_indices <= Ny/2, j_indices * delky, (j_indices - Ny) * delky)
    # If j <= Ny/2, ky = j*delky, else ky = (j-Ny)*delky

    # Calculate k2 and k4
    # kx[:, cun.newaxis] will reshape it to (Nx, 1)
    # ky[cun.newaxis, :] will reshape it to (1, Ny)
    # Their squared addition will result in a (Nx, Ny) matrix
    # k2[first row] = [kx[0]**2 + ky[0]**2, kx[0]**2 + ky[1]**2, ..., kx[0]**2 + ky[Ny-1]**2]
    k2 = kx[:, cun.newaxis]**2 + ky[cun.newaxis, :]**2
    k4 = k2**2

    # Outer iterations of 1000
    for m in range(num_iter):
        # Inner iterations of 100
        for n in range(steps_per_iter):
            
            #g stores the first derivative of free energy df/dc = 2*A*c*(1-c)*(1-2c)
            mult = cun.multiply(1-cnew,1-2*cnew)
            g = 2*A*cun.multiply(cnew,mult)
            ghat = cun.fft.fft2(g)
            chat = cun.fft.fft2(cnew)
            
            chat = (chat - M*dt*k2*ghat)/(1+2*M*kappa*k4*dt)
                    
            cnew = cun.fft.ifft2(chat).real
            c = cnew

        if verbose: print(m)

    end_time = time("s")

    if verbose:
        clear_output(wait=True)
        X,Y = np.meshgrid(range(Nx),range(Ny))        
        plt.contourf(X,Y,c,cmap = 'jet')
        plt.colorbar()
        plt.show()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    if verbose:
        print("Execution Time:", execution_time, "seconds")
    return execution_time


def plot_graph(times, labels, sizes, title,
               x_lim = None,
               y_lim = None,
               show_matrix_size = False):
    for i, time_taken in enumerate(times):
        plt.plot(sizes, time_taken, label=labels[i])
    plt.xlabel('Array Size')
    plt.ylabel('Time (s)')
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    if show_matrix_size:
        # In x-axis, display the array sizes as 250x250, 500x500, etc.
        plt.xticks(sizes, [f'{size}x{size}' for size in sizes])
    plt.legend()
    plt.grid()
    plt.show()

def main():
    arr_sizes = [16, 32, 64, 128, 200, 256, 512, 768, 1024]
    NUM_ITER = 8
    # Using numpy
    np_times = []
    for size in tqdm(arr_sizes, desc="Numpy"):
        time_taken = np_spinodal(arr_size=size, num_iter = NUM_ITER, 
                                verbose=False)
        np_times.append(time_taken)

    # Using scipy
    sp_times = []
    for size in tqdm(arr_sizes, desc="Scipy"):
        time_taken = sp_spinodal(arr_size=size, num_iter = NUM_ITER, 
                                verbose=False)
        sp_times.append(time_taken)

    # Using pyfftw
    pyfftw_times = []
    for size in tqdm(arr_sizes, desc="PyFFTW"):
        time_taken = pyfftw_spinodal(arr_size=size, num_iter = NUM_ITER, 
                                    verbose=False)
        pyfftw_times.append(time_taken)


    # Using cunumeric
    cun_times = []
    for size in tqdm(arr_sizes, desc="CuNumeric"):
        time_taken = cun_spinodal(arr_size=size, num_iter = NUM_ITER, 
                                verbose=False)
        cun_times.append(time_taken)

    labels = ["Numpy", "Scipy", "PyFFTW", "CuNumeric"]
    times = [np_times, sp_times, pyfftw_times, cun_times]
    title = "Spinodal-2D with different libraries"
    plot_graph(times, labels, arr_sizes, title)

if __name__ == "__main__":
    main()
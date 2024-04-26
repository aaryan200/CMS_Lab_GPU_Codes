import numpy as np
from legate.timing import time
import cunumeric as cu
import scipy as sc
import pyfftw
from tqdm import tqdm
import matplotlib.pyplot as plt

def ternary_numpy(n, num_iter = 2000):
    # Grid dimensions and constants
    Nx, Ny = n, n
    dx, dy = 1.0, 1.0  # Spatial resolution
    dt = 0.1  # Time step

    # Initialize concentrations
    cA = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cB = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cC = 1 - cA - cB

    start_time = time("s")

    # Create wave number arrays adjusted for rfft2
    kx = 2 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Constants for the differential operators and reaction parameters
    M_AA, M_BB, M_AB = 1, 1, 0.5
    kA, kB, kC = 1.0, 1.0, 1.0
    kAA = kA + kC
    kBB = kB + kC
    kAB = kC
    A1, A2, A3, B = 1.0, 1.0, 1.0, 12.0

    # Precompute lhs
    lhsA = 1 + 2 * dt * k4 * (M_AA * kAA - M_AB * kAB)
    lhsB = 1 + 2 * dt * k4 * (M_BB * kBB - M_AB * kAB)

    # Coordinates
    x = np.arange(0, Nx) * dx
    y = np.arange(0, Ny) * dy
    z = np.zeros(1)  # Since this is a 2D data set

    # Simulation loop
    for n in range(num_iter):
        gA = 2 * A1 * cA * cB**2 - 2 * A2 * cB**2 * cC - 2 * A3 * cA**2 * cC + 2 * A3 * cA * cC**2 - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA * cB**2 * cC**2
        gB = 2 * A1 * cA**2 * cB - 2 * A2 * cB**2 * cC + 2 * A2 * cB * cC**2 - 2 * A3 * cA**2 * cC - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA**2 * cB * cC**2
        
        # Fourier transforms
        cA_tilda = np.fft.rfft2(cA)
        cB_tilda = np.fft.rfft2(cB)
        gA_tilda = np.fft.rfft2(gA)
        gB_tilda = np.fft.rfft2(gB)

        # Update in Fourier space
        cA_tilda = (cA_tilda - k2 * dt * (M_AA * gA_tilda - M_AB * gB_tilda) - 2 * k4 * dt * cB_tilda * (M_AA * kAB - M_AB * kBB)) / lhsA
        cB_tilda = (cB_tilda - k2 * dt * (M_BB * gB_tilda - M_AB * gA_tilda) - 2 * k4 * dt * cA_tilda * (M_BB * kAB - M_AB * kAA)) / lhsB

        # Inverse Fourier transforms to update concentrations
        cA = np.fft.irfft2(cA_tilda)
        cB = np.fft.irfft2(cB_tilda)
        cC = 1 - cA - cB

        # Clip concentrations
        cA = np.clip(cA, 0, 1)
        cB = np.clip(cB, 0, 1)
        cC = np.clip(cC, 0, 1)

    # Compute microstructure
    microstruct = np.zeros((Nx, Ny))
    microstruct = np.where((cA < 0.5) & (cB < 0.5) & (cC < 0.5), 0, microstruct)
    microstruct = np.where(cA > 0.5, cA, microstruct)
    microstruct = np.where(cB > 0.5, 0.65 * cB, microstruct)
    microstruct = np.where(cC > 0.5, 0.35 * cC, microstruct)

    end_time = time("s")
    execution_time  = end_time - start_time
    return execution_time

def ternary_cunumeric_naive(n, num_iter = 2000):
    '''
    Naive method: Replace numpy with cunumeric.
    '''
    # Grid dimensions and constants
    Nx, Ny = n, n
    dx, dy = 1.0, 1.0  # Spatial resolution
    dt = 0.1  # Time step

    # Initialize concentrations
    cA = (1.0 / 3) + cu.random.normal(0, 0.001, (Nx, Ny))
    cB = (1.0 / 3) + cu.random.normal(0, 0.001, (Nx, Ny))
    cC = 1 - cA - cB

    start_time = time("s")

    # Create wave number arrays adjusted for rfft2
    kx = 2 * cu.pi * cu.fft.rfftfreq(Nx, d=dx)
    ky = 2 * cu.pi * cu.fft.fftfreq(Ny, d=dy)
    kx, ky = cu.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Constants for the differential operators and reaction parameters
    M_AA, M_BB, M_AB = 1, 1, 0.5
    kA, kB, kC = 1.0, 1.0, 1.0
    kAA = kA + kC
    kBB = kB + kC
    kAB = kC
    A1, A2, A3, B = 1.0, 1.0, 1.0, 12.0

    # Precompute lhs
    lhsA = 1 + 2 * dt * k4 * (M_AA * kAA - M_AB * kAB)
    lhsB = 1 + 2 * dt * k4 * (M_BB * kBB - M_AB * kAB)

    # Coordinates
    x = cu.arange(0, Nx) * dx
    y = cu.arange(0, Ny) * dy
    z = cu.zeros(1)  # Since this is a 2D data set

    # Simulation loop
    for n in range(num_iter):
        gA = 2 * A1 * cA * cB**2 - 2 * A2 * cB**2 * cC - 2 * A3 * cA**2 * cC + 2 * A3 * cA * cC**2 - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA * cB**2 * cC**2
        gB = 2 * A1 * cA**2 * cB - 2 * A2 * cB**2 * cC + 2 * A2 * cB * cC**2 - 2 * A3 * cA**2 * cC - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA**2 * cB * cC**2
        
        # Fourier transforms
        cA_tilda = cu.fft.rfft2(cA)
        cB_tilda = cu.fft.rfft2(cB)
        gA_tilda = cu.fft.rfft2(gA)
        gB_tilda = cu.fft.rfft2(gB)

        # Update in Fourier space
        cA_tilda = (cA_tilda - k2 * dt * (M_AA * gA_tilda - M_AB * gB_tilda) - 2 * k4 * dt * cB_tilda * (M_AA * kAB - M_AB * kBB)) / lhsA
        cB_tilda = (cB_tilda - k2 * dt * (M_BB * gB_tilda - M_AB * gA_tilda) - 2 * k4 * dt * cA_tilda * (M_BB * kAB - M_AB * kAA)) / lhsB

        # Inverse Fourier transforms to update concentrations
        cA = cu.fft.irfft2(cA_tilda)
        cB = cu.fft.irfft2(cB_tilda)
        cC = 1 - cA - cB

        # Clip concentrations
        cA = cu.clip(cA, 0, 1)
        cB = cu.clip(cB, 0, 1)
        cC = cu.clip(cC, 0, 1)

    # Compute microstructure
    microstruct = cu.zeros((Nx, Ny))
    microstruct = cu.where((cA < 0.5) & (cB < 0.5) & (cC < 0.5), 0, microstruct)
    microstruct = cu.where(cA > 0.5, cA, microstruct)
    microstruct = cu.where(cB > 0.5, 0.65 * cB, microstruct)
    microstruct = cu.where(cC > 0.5, 0.35 * cC, microstruct)
    
    end_time = time("s")
    execution_time  = end_time - start_time
    return execution_time

def ternary_cunumeric_optimized(n, num_iter = 2000):
    '''
    Optimized method: Reduce the number of operations by combining the calculations
    and using the cu.multiply and cu.add functions.
    '''
    # Grid dimensions and constants
    Nx, Ny = n, n
    dx, dy = 1.0, 1.0  # Spatial resolution
    dt = 0.1  # Time step

    # Initialize concentrations
    cA = (1.0 / 3) + cu.random.normal(0, 0.001, (Nx, Ny))
    cB = (1.0 / 3) + cu.random.normal(0, 0.001, (Nx, Ny))
    cC = 1 - cA - cB

    start_time = time("s")

    # Create wave number arrays adjusted for rfft2
    kx = 2 * cu.pi * cu.fft.rfftfreq(Nx, d=dx)
    ky = 2 * cu.pi * cu.fft.fftfreq(Ny, d=dy)
    kx, ky = cu.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Constants for the differential operators and reaction parameters
    M_AA, M_BB, M_AB = 1, 1, 0.5
    kA, kB, kC = 1.0, 1.0, 1.0
    kAA = kA + kC
    kBB = kB + kC
    kAB = kC
    A1, A2, A3, B = 1.0, 1.0, 1.0, 12.0

    # Precompute lhs
    lhsA = 1 + 2 * dt * k4 * (M_AA * kAA - M_AB * kAB)
    lhsB = 1 + 2 * dt * k4 * (M_BB * kBB - M_AB * kAB)

    # Coordinates
    x = cu.arange(0, Nx) * dx
    y = cu.arange(0, Ny) * dy
    z = cu.zeros(1)  # Since this is a 2D data set

    # Simulation loop
    for n in range(num_iter):
        gA = cu.multiply(2*cB**2, A1*cA - A2 * cC) + cu.multiply(2*A3*cA*cC, cC - cA) + cu.multiply(2*B*cA*(cB**2)*cC, cC - cA)
        gB = cu.multiply(2*cA**2, A1*cB - A3*cC) + cu.multiply(2*A2*cB*cC, cC - cB) + cu.multiply(2*B*(cA**2)*cB*cC, cC - cB)
        
        # Fourier transforms
        cA_tilda = cu.fft.rfft2(cA)
        cB_tilda = cu.fft.rfft2(cB)
        gA_tilda = cu.fft.rfft2(gA)
        gB_tilda = cu.fft.rfft2(gB)

        # Update in Fourier space
        cA_tilda = (cA_tilda - k2 * dt * (M_AA * gA_tilda - M_AB * gB_tilda) - 2 * k4 * dt * cB_tilda * (M_AA * kAB - M_AB * kBB)) / lhsA
        cB_tilda = (cB_tilda - k2 * dt * (M_BB * gB_tilda - M_AB * gA_tilda) - 2 * k4 * dt * cA_tilda * (M_BB * kAB - M_AB * kAA)) / lhsB

        # Inverse Fourier transforms to update concentrations
        cA = cu.fft.irfft2(cA_tilda)
        cB = cu.fft.irfft2(cB_tilda)
        cC = cu.subtract(1, cu.add(cA, cB))

        # Clip concentrations
        cA = cu.clip(cA, 0, 1)
        cB = cu.clip(cB, 0, 1)
        cC = cu.clip(cC, 0, 1)

    # Compute microstructure
    microstruct = cu.zeros((Nx, Ny))
    microstruct = cu.where((cA < 0.5) & (cB < 0.5) & (cC < 0.5), 0, microstruct)
    microstruct = cu.where(cA > 0.5, cA, microstruct)
    microstruct = cu.where(cB > 0.5, 0.65 * cB, microstruct)
    microstruct = cu.where(cC > 0.5, 0.35 * cC, microstruct)

    end_time = time("s")
    execution_time  = end_time - start_time
    return execution_time

def ternary_scipy(n, num_iter = 2000):
    '''
    Uses scipy's fft.rfftfreq, fft.fftfreq, fft.rfft2 and fft.irfft2
    Others are all same as numpy
    '''
    # Grid dimensions and constants
    Nx, Ny = n, n
    dx, dy = 1.0, 1.0  # Spatial resolution
    dt = 0.1  # Time step

    # Initialize concentrations
    cA = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cB = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cC = 1 - cA - cB

    start_time = time("s")

    # Create wave number arrays adjusted for rfft2
    kx = 2 * np.pi * sc.fft.rfftfreq(Nx, d=dx)
    ky = 2 * np.pi * sc.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Constants for the differential operators and reaction parameters
    M_AA, M_BB, M_AB = 1, 1, 0.5
    kA, kB, kC = 1.0, 1.0, 1.0
    kAA = kA + kC
    kBB = kB + kC
    kAB = kC
    A1, A2, A3, B = 1.0, 1.0, 1.0, 12.0

    # Precompute lhs
    lhsA = 1 + 2 * dt * k4 * (M_AA * kAA - M_AB * kAB)
    lhsB = 1 + 2 * dt * k4 * (M_BB * kBB - M_AB * kAB)

    # Coordinates
    x = np.arange(0, Nx) * dx
    y = np.arange(0, Ny) * dy
    z = np.zeros(1)  # Since this is a 2D data set

    # Simulation loop
    for _ in range(num_iter):
        gA = 2 * A1 * cA * cB**2 - 2 * A2 * cB**2 * cC - 2 * A3 * cA**2 * cC + 2 * A3 * cA * cC**2 - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA * cB**2 * cC**2
        gB = 2 * A1 * cA**2 * cB - 2 * A2 * cB**2 * cC + 2 * A2 * cB * cC**2 - 2 * A3 * cA**2 * cC - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA**2 * cB * cC**2

        # Fourier transforms
        cA_tilda = sc.fft.rfft2(cA)
        cB_tilda = sc.fft.rfft2(cB)
        gA_tilda = sc.fft.rfft2(gA)
        gB_tilda = sc.fft.rfft2(gB)
        

        # Update in Fourier space
        cA_tilda = (cA_tilda - k2 * dt * (M_AA * gA_tilda - M_AB * gB_tilda) - 2 * k4 * dt * cB_tilda * (M_AA * kAB - M_AB * kBB)) / lhsA
        cB_tilda = (cB_tilda - k2 * dt * (M_BB * gB_tilda - M_AB * gA_tilda) - 2 * k4 * dt * cA_tilda * (M_BB * kAB - M_AB * kAA)) / lhsB

        # Inverse Fourier transforms to update concentrations
        cA = sc.fft.irfft2(cA_tilda)
        cB = sc.fft.irfft2(cB_tilda)
        cC = 1 - cA - cB

        # Clip concentrations
        cA = np.clip(cA, 0, 1)
        cB = np.clip(cB, 0, 1)
        cC = np.clip(cC, 0, 1)
    
    # Compute microstructure
    microstruct = np.zeros((Nx, Ny))
    microstruct = np.where((cA < 0.5) & (cB < 0.5) & (cC < 0.5), 0, microstruct)
    microstruct = np.where(cA > 0.5, cA, microstruct)
    microstruct = np.where(cB > 0.5, 0.65 * cB, microstruct)
    microstruct = np.where(cC > 0.5, 0.35 * cC, microstruct)

    end_time = time("s")
    execution_time = end_time - start_time
    return execution_time

def ternary_pyfftw(n, num_iter = 2000):
    '''
    Other than rfft2 and irfft2, all are numpy library
    '''
    # Grid dimensions and constants
    Nx, Ny = n, n
    dx, dy = 1.0, 1.0  # Spatial resolution
    dt = 0.1  # Time step

    # Initialize concentrations
    cA = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cB = (1.0 / 3) + np.random.normal(0, 0.001, (Nx, Ny))
    cC = 1 - cA - cB

    start_time = time("s")

    # Create wave number arrays adjusted for rfft2
    kx = 2 * np.pi * np.fft.rfftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k4 = k2**2

    # Constants for the differential operators and reaction parameters
    M_AA, M_BB, M_AB = 1, 1, 0.5
    kA, kB, kC = 1.0, 1.0, 1.0
    kAA = kA + kC
    kBB = kB + kC
    kAB = kC
    A1, A2, A3, B = 1.0, 1.0, 1.0, 12.0

    # Precompute lhs
    lhsA = 1 + 2 * dt * k4 * (M_AA * kAA - M_AB * kAB)
    lhsB = 1 + 2 * dt * k4 * (M_BB * kBB - M_AB * kAB)

    # Coordinates
    x = np.arange(0, Nx) * dx
    y = np.arange(0, Ny) * dy
    z = np.zeros(1)  # Since this is a 2D data set

    # Simulation loop
    for n in range(num_iter):
        gA = 2 * A1 * cA * cB**2 - 2 * A2 * cB**2 * cC - 2 * A3 * cA**2 * cC + 2 * A3 * cA * cC**2 - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA * cB**2 * cC**2
        gB = 2 * A1 * cA**2 * cB - 2 * A2 * cB**2 * cC + 2 * A2 * cB * cC**2 - 2 * A3 * cA**2 * cC - 2 * B * cA**2 * cB**2 * cC + 2 * B * cA**2 * cB * cC**2
        
        # Fourier transforms
        cA_tilda = pyfftw.interfaces.numpy_fft.rfft2(cA)
        cB_tilda = pyfftw.interfaces.numpy_fft.rfft2(cB)
        gA_tilda = pyfftw.interfaces.numpy_fft.rfft2(gA)
        gB_tilda = pyfftw.interfaces.numpy_fft.rfft2(gB)

        # Update in Fourier space
        cA_tilda = (cA_tilda - k2 * dt * (M_AA * gA_tilda - M_AB * gB_tilda) - 2 * k4 * dt * cB_tilda * (M_AA * kAB - M_AB * kBB)) / lhsA
        cB_tilda = (cB_tilda - k2 * dt * (M_BB * gB_tilda - M_AB * gA_tilda) - 2 * k4 * dt * cA_tilda * (M_BB * kAB - M_AB * kAA)) / lhsB

        # Inverse Fourier transforms to update concentrations
        cA = pyfftw.interfaces.numpy_fft.irfft2(cA_tilda)
        cB = pyfftw.interfaces.numpy_fft.irfft2(cB_tilda)
        cC = 1 - cA - cB

        # Clip concentrations
        cA = np.clip(cA, 0, 1)
        cB = np.clip(cB, 0, 1)
        cC = np.clip(cC, 0, 1)

    # Compute microstructure
    microstruct = np.zeros((Nx, Ny))
    microstruct = np.where((cA < 0.5) & (cB < 0.5) & (cC < 0.5), 0, microstruct)
    microstruct = np.where(cA > 0.5, cA, microstruct)
    microstruct = np.where(cB > 0.5, 0.65 * cB, microstruct)
    microstruct = np.where(cC > 0.5, 0.35 * cC, microstruct)

    end_time = time("s")
    execution_time = end_time - start_time
    return execution_time
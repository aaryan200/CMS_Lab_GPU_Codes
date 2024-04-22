import cunumeric as np1
import numpy as np
import time
N = 10000
n = int(np.sqrt(N))
start1 = time.time()
a = np.arange(N, dtype = int)
b = np.arange(N, dtype = int)
a = np.reshape(a,(n,n))
b = np.reshape(b,(n,n))
c = np.multiply(a,b)
stop1 = time.time()
start2 = time.time()
a1 = np1.arange(N,dtype= int)
b1 = np1.arange(N,dtype= int)
a1 = np1.reshape(a1,(n,n))
b1 = np1.reshape(b1,(n,n))
c1 = np1.multiply(a,b)
stop2 = time.time()
print(f"time taken(NumPy):{(-start1+stop1)*1000} ms")
print(f"time taken(CuNumeric):{(-start2+stop2)*1000} ms")


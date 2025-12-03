import numpy as np

n = 120
num = 0
n1 = round(0.04*n+4.41)
alfa = np.pi/n1
for j in range(n1+1):
    lat = alfa*j
    n2 = round(2*np.sin(j*alfa)*n1) + 1
    Dl = 2*np.pi/n2
    for i in range (n2):
        num +=1
        long = i*Dl
        print(round(lat*180/np.pi), round(long*180/np.pi))
print(num)


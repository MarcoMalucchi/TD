import tdwf
import numpy as np
import matplotlib.pyplot as plt
import time

plt.close('all')

ad2 = tdwf.AD2()
scope = tdwf.Scope(ad2.hdwf)
scope.fs = 1e2
scope.npt = 500

scope.ch1.range = 5.0
scope.ch2.range = 50.0
#scope.ch1.avg = True

scope.sample()
time.sleep(1)
print(scope.ch1.vals)

plt.plot(scope.time.vals, scope.ch1.vals, '.')
plt.xlabel('tempo [s]')
plt.ylabel('Volt [V]')

plt.show()

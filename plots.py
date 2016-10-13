import matplotlib.pyplot as plt
import numpy as np

basis = 'mod'

err = np.loadtxt('error_{}.dat'.format(basis))
it = np.loadtxt('it_{}.dat'.format(basis))
errk = np.loadtxt('errk_{}.dat'.format(basis))

print err
plt.semilogy(range(26,-1, -1), err, label='$\phi$')
plt.semilogy(range(26,-1, -1), errk, label='k')
plt.legend()
plt.show()
plt.clf()

plt.plot(range(26,-1, -1), it)
plt.show()
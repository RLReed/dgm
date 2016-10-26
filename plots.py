import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
rc('font',**{'family':'serif'})
from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.labelsize'] = 18
rcParams['ytick.labelsize'] = 18
rcParams['lines.linewidth'] = 1.85
rcParams['axes.labelsize'] = 20
rcParams.update({'figure.autolayout': True})

def fixedPlots():
    G = 238
    dlp = np.loadtxt('fixedDiv4_dlp{}.dat'.format(G))
    mdlp = np.loadtxt('fixedDiv4_mdlp{}.dat'.format(G))
    pwr = np.loadtxt('fixedDiv4_mdlp{}Pwr.dat'.format(G))
    triga = np.loadtxt('fixedDiv4_mdlp{}Triga.dat'.format(G))
    mox = np.loadtxt('fixedDiv4_mdlp{}MOX.dat'.format(G))
    x = range(50)
    
    plt.semilogy(x, dlp, 'k-', label='DLP')
    plt.semilogy(x, mdlp, 'b--', label='mDLP')
    plt.semilogy(x, pwr, 'g:', label='PWR')
    plt.semilogy(x, triga, 'r-', label='Triga')
    plt.semilogy(x, mox, 'm-', label='MOX')
    plt.legend(loc=0)
    plt.ylabel('Relative error in $L_2$ norm')
    plt.xlabel('Degrees of freedom in basis')
    plt.savefig('fixedDiv4_comparison.pdf')
    plt.show()
    

def otherThing():
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
    
if __name__ == "__main__":
    fixedPlots()

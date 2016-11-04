from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py
# Make plots look nicer
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
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k', 'r', 'g', 'b', 'y', 'k', 'r', 'g', 'b', 'y', 'k', 'r', 'g', 'b', 'y', 'k']) +
                           cycler('linestyle', ['-', '-', '-', '-', '-', '--', '--', '--', '--', '--', ':', ':', ':', ':', ':', '-.', '-.', '-.', '-.', '-.'])))

class fixedSourceSolver(object):
    def __init__(self, sig_t, sig_s, Q, phi=None, lamb=1, truncate=None, bType='dlp', bPattern=None, division=None):
        self.sig_t = np.array(sig_t)
        self.Q = Q
        self.sig_s = np.array(sig_s)
        self.phi = np.ones(len(sig_t)) if phi is None else phi
        self.lamb = lamb
        
        # Number of fine groups
        self.nFine = len(sig_t)
        
        # Determine course group structure
        self.courseStruct = [0]
        if division is not None:
            assert max(division) < self.nFine and min(division) > 0, "Group division outside of available groups"
            self.courseStruct = [0] + division
            course = np.array(self.courseStruct + [self.nFine])
            self.fineStruct = course[1:] - course[:-1]
        else:
            self.fineStruct = [self.nFine]
            
        # Number of course groups
        self.nCourse = len(self.courseStruct)
        
        self.bounds = [0] + np.cumsum(self.fineStruct).tolist()
        
        self.basis = []
        self.coef = []
        for G, groups in enumerate(self.fineStruct):
            # Get the basis for the problem
            B = getBasis(bType, groups, bPattern[self.bounds[G]:self.bounds[G+1]])
            trunc = self.nFine if truncate is None else truncate
            self.basis.append(B.basis[:trunc+1])
            self.coef.append(B.coef[:trunc+1])
        
    def calcDGMCrossSections(self):
        def getPhiCoef(phi, basis):
            return basis.dot(phi)
        
        def getTotal(sig_t, phi, basis):
            return basis[0].dot(np.diag(sig_t)).dot(phi) / basis[0].dot(phi)
        
        def getDelta(sig_t, T, phi, basis):
            return basis.dot(np.diag(sig_t - T).dot(phi)) / basis[0].dot(phi)
        
        def getScatter(sig_s, phi, basis, phi0):
            return basis.dot(sig_s).dot(phi) / phi0
        
        def getSource(basis, Q):
            return basis.dot(Q)
        
        self.Pa = []
        self.Ta = []
        self.da = []
        self.Sa = []
        self.Qa = []
        for G in range(self.nCourse):
            # Set reused parameters
            T = self.sig_t[self.bounds[G]:self.bounds[G+1]]
            phiG = self.phi[self.bounds[G]:self.bounds[G+1]]
            # Get expanded phi
            self.Pa.append(getPhiCoef(phiG, self.basis[G]))
            # Get expanded source
            self.Qa.append(getSource(self.basis[G], self.Q[self.bounds[G]:self.bounds[G+1]]))
            # Get total DGM cross section
            self.Ta.append(getTotal(T, phiG, self.basis[G]))
            # Get delta DGM cross section
            self.da.append(getDelta(T, self.Ta[G], phiG, self.basis[G]))
            # Get Scattering DGM cross section
            Sa = []
            for Gp in range(self.nCourse):
                sig_s = self.sig_s[self.bounds[G]:self.bounds[G+1], self.bounds[Gp]:self.bounds[Gp+1]]
                phi = self.phi[self.bounds[Gp]:self.bounds[Gp+1]]
                phi0 = self.basis[Gp][0].dot(phi)
                Sa.append(getScatter(sig_s, phi, self.basis[G], phi0))
            self.Sa.append(Sa)
        
    def update(self):
        # Loop over course groups
        for G in range(self.nCourse):
            S = 0
            for Gp in range(self.nCourse):
                S += self.Sa[G][Gp] * self.Pa[Gp][0]
            self.Pa[G] = (S + self.Qa[G] - self.da[G] * self.Pa[G][0]) / self.Ta[G]
        phi = []
        for G in range(self.nCourse):
            phi.append(self.basis[G].T.dot(self.Pa[G]))
        phi = np.concatenate(phi)
        
        self.phi = (1 - self.lamb) * self.phi + self.lamb * phi
        
    def solve(self, analytic=None, silent=False):
        ep = 1
        i = 0
        if not silent:
            print i, ep#, self.phi
        while ep > 1e-12:
            old = self.phi.copy()
            self.calcDGMCrossSections()
            self.update()
            if analytic is None:
                ep = np.linalg.norm(old-self.phi)
            else:
                ep = np.linalg.norm(analytic-self.phi)
            i += 1
            if not silent:
                print i, ep#, self.phi
            if i > 10000 and False:
                print 'WARNING: iteration not converged'
                break
        return self.phi
    
class getBasis(object):
    def __init__(self, btype, G, pattern=None):
        if btype.lower() == 'dlp':
            self.DLP(G)
        if btype.lower() == 'mdlp':
            self.mDLP(G, pattern)
            
    def DLP(self, G):
        basis = np.zeros((G, G))
        for i in range(G):
            basis[i] = np.polynomial.Legendre.basis(i)(np.linspace(-1,1,G))
        self.basis = np.linalg.qr(basis.T, 'full')[0].T
        self.coef = np.diag(basis.T.dot(basis))
        
    def mDLP(self, G, pattern):
        assert pattern is not None
        self.DLP(G)
        basis = self.basis.copy()
        for i in range(G):
            basis[i] *= pattern
            
        self.basis[1:] = basis[:-1]
        #self.basis = basis
        
        self.basis = np.linalg.qr(self.basis.T, 'full')[0].T
        self.coef = np.diag(self.basis.T.dot(self.basis))
        
def getData(groups):
    if groups == 2:
        sig_t = np.array([1,2])
        sig_s = np.array([[0.3, 0],
                          [0.3, 0.3]])
        S = np.array([1, 0.000000000])
        division = None
        division = [1]
    
    elif groups == 3:
        sig_t = np.array([0.2822058997,0.4997685502,0.4323754911])
        sig_s = np.array([[0.2760152893, 0.0000000000, 0.0000000000],
                          [0.0011230014, 0.4533430274, 0.0000378305],
                          [0.0000000000, 0.0014582502, 0.2823864370]])
        S = np.array([0.9996892490, 0.0003391680, 0.000000000])
        division = None
        division = [2]
    
    elif groups == 7:
        sig_t = np.array([0.21245,0.355470, 0.48554, 0.5594, 0.31803, 0.40146, 0.57061])
        sig_s = np.array([[1.27537e-1, 4.2378e-2, 9.4374e-6, 5.5163e-9, 0, 0, 0],
                          [0, 3.24456e-1, 1.6314e-3, 3.1427e-9, 0,0,0],
                          [0, 0, 4.5094e-1, 2.6792e-3, 0, 0, 0],
                          [0, 0, 0, 4.52565e-1, 5.5664e-3, 0, 0],
                          [0, 0, 0, 1.2525e-4, 2.71401e-1, 1.0255e-2, 1.0021e-8],
                          [0, 0, 0, 0, 1.2968e-3, 2.65802e-1, 1.6809e-2],
                          [0, 0, 0, 0, 0, 8.5458e-3, 2.7308e-1]]).T
        S = np.array([5.87910e-1, 4.1176e-1, 3.3906e-4, 1.1761e-7, 0, 0, 0])
        division = [4]
    
    elif groups == 238:
        # 238 group data from c5g7 geometry
        # Using UO2 data
        with h5py.File('uo2XS238-5pct.h5', 'r') as f:
            sig_t = f['sig_t'][...]
            sig_s = f['sig_s'][...]
            S = f['chi'][...]
            #S = np.ones(groups)
        division = [50, 100, 150, 200]
        #division = [236]
    else:
        raise ValueError, "Group not yet implemented"
            
    return sig_t, sig_s, S, division

def getMax(x):
    x = np.concatenate(x, [len(x)])
    x = np.concatenate([0], x)
    return x[1:] - x[:-1]

def plotPhi(phi):
    E = np.array([2.00000000e+07,1.73330000e+07,1.56830000e+07,1.45500000e+07,1.38400000e+07,1.28400000e+07,
         1.00000000e+07,8.18730000e+06,6.43400000e+06,4.80000000e+06,4.30400000e+06,3.00000000e+06,
         2.47900000e+06,2.35400000e+06,1.85000000e+06,1.50000000e+06,1.40000000e+06,1.35600000e+06,
         1.31700000e+06,1.25000000e+06,1.20000000e+06,1.10000000e+06,1.01000000e+06,9.20000000e+05,
         9.00000000e+05,8.75000000e+05,8.61100000e+05,8.20000000e+05,7.50000000e+05,6.79000000e+05,
         6.70000000e+05,6.00000000e+05,5.73000000e+05,5.50000000e+05,4.99520000e+05,4.70000000e+05,
         4.40000000e+05,4.20000000e+05,4.00000000e+05,3.30000000e+05,2.70000000e+05,2.00000000e+05,
         1.50000000e+05,1.28300000e+05,1.00000000e+05,8.50000000e+04,8.20000000e+04,7.50000000e+04,
         7.30000000e+04,6.00000000e+04,5.20000000e+04,5.00000000e+04,4.50000000e+04,3.00000000e+04,
         2.50000000e+04,1.70000000e+04,1.30000000e+04,9.50000000e+03,8.03000000e+03,6.00000000e+03,
         3.90000000e+03,3.74000000e+03,3.00000000e+03,2.58000000e+03,2.29000000e+03,2.20000000e+03,
         1.80000000e+03,1.55000000e+03,1.50000000e+03,1.15000000e+03,9.50000000e+02,6.83000000e+02,
         6.70000000e+02,5.50000000e+02,3.05000000e+02,2.85000000e+02,2.40000000e+02,2.10000000e+02,
         2.07500000e+02,1.92500000e+02,1.86000000e+02,1.22000000e+02,1.19000000e+02,1.15000000e+02,
         1.08000000e+02,1.00000000e+02,9.00000000e+01,8.20000000e+01,8.00000000e+01,7.60000000e+01,
         7.20000000e+01,6.75000000e+01,6.50000000e+01,6.10000000e+01,5.90000000e+01,5.34000000e+01,
         5.20000000e+01,5.06000000e+01,4.92000000e+01,4.83000000e+01,4.70000000e+01,4.52000000e+01,
         4.40000000e+01,4.24000000e+01,4.10000000e+01,3.96000000e+01,3.91000000e+01,3.80000000e+01,                
         3.70000000e+01,3.55000000e+01,3.46000000e+01,3.37500000e+01,3.32500000e+01,3.17500000e+01,
         3.12500000e+01,3.00000000e+01,2.75000000e+01,2.50000000e+01,2.25000000e+01,2.10000000e+01,
         2.00000000e+01,1.90000000e+01,1.85000000e+01,1.70000000e+01,1.60000000e+01,1.51000000e+01,
         1.44000000e+01,1.37500000e+01,1.29000000e+01,1.19000000e+01,1.15000000e+01,1.00000000e+01,
         9.10000000e+00,8.10000000e+00,7.15000000e+00,7.00000000e+00,6.75000000e+00,6.50000000e+00,
         6.25000000e+00,6.00000000e+00,5.40000000e+00,5.00000000e+00,4.75000000e+00,4.00000000e+00,
         3.73000000e+00,3.50000000e+00,3.15000000e+00,3.05000000e+00,3.00000000e+00,2.97000000e+00,
         2.87000000e+00,2.77000000e+00,2.67000000e+00,2.57000000e+00,2.47000000e+00,2.38000000e+00,
         2.30000000e+00,2.21000000e+00,2.12000000e+00,2.00000000e+00,1.94000000e+00,1.86000000e+00,
         1.77000000e+00,1.68000000e+00,1.59000000e+00,1.50000000e+00,1.45000000e+00,1.40000000e+00,
         1.35000000e+00,1.30000000e+00,1.25000000e+00,1.22500000e+00,1.20000000e+00,1.17500000e+00,
         1.15000000e+00,1.14000000e+00,1.13000000e+00,1.12000000e+00,1.11000000e+00,1.10000000e+00,
         1.09000000e+00,1.08000000e+00,1.07000000e+00,1.06000000e+00,1.05000000e+00,1.04000000e+00,
         1.03000000e+00,1.02000000e+00,1.01000000e+00,1.00000000e+00,9.75000000e-01,9.50000000e-01,
         9.25000000e-01,9.00000000e-01,8.50000000e-01,8.00000000e-01,7.50000000e-01,7.00000000e-01,
         6.50000000e-01,6.25000000e-01,6.00000000e-01,5.50000000e-01,5.00000000e-01,4.50000000e-01,
         4.00000000e-01,3.75000000e-01,3.50000000e-01,3.25000000e-01,3.00000000e-01,2.75000000e-01,
         2.50000000e-01,2.25000000e-01,2.00000000e-01,1.75000000e-01,1.50000000e-01,1.25000000e-01,
         1.00000000e-01,9.00000000e-02,8.00000000e-02,7.00000000e-02,6.00000000e-02,5.00000000e-02,
         4.00000000e-02,3.00000000e-02,2.53000000e-02,1.00000000e-02,7.50000000e-03,5.00000000e-03,
         4.00000000e-03,3.00000000e-03,2.50000000e-03,2.00000000e-03,1.50000000e-03,1.20000000e-03,
         1.00000000e-03,7.50000000e-04,5.00000000e-04,1.00000000e-04,1.00000000e-05])
    E0 = E[:-1]
    E1 = E[1:]
    de = E[:-1] - E[1:]
    E = np.concatenate((E0, E1)).reshape(-1,len(E0)).T.flatten()
    phi = np.concatenate((phi / de, phi / de)).reshape(-1,len(E0)).T.flatten()
    
    plt.loglog(E, phi)
    plt.ylabel('$\phi$')
    plt.xlabel('Energy [eV]')
    plt.show()
    
def runFracUO2():
    # 238 group data from UO2 pin cell geometry
    with h5py.File('uo2XS-0.3.h5', 'r') as f:
        sig_t = f['sig_t'][...]
        sig_s = f['sig_s'][...]
        S = f['chi'][...]
        #S = np.ones(groups)
        p = f['phi'][...]
    div = [50, 100, 150, 200]
    #division = [236]
    
    M = np.diag(sig_t) - sig_s
    print 'Fixed Source Problem'
    phi = np.linalg.solve(M, S)
    print p - phi
    asdf
    print phi
    plotPhi(phi)
    
    for f in np.linspace(0,1,21):
        with h5py.File('uo2XS-{}.h5'.format(f), 'r') as F:
            pat = F['phi'][...]
            
        bType = 'mdlp'
        eps = []
        x = range(50)
        for trunc in x:
            P = fixedSourceSolver(sig_t, sig_s, S, lamb=0.05, truncate=trunc, bType=bType, bPattern=pat, division=div).solve(silent=True)
            
            print 'Fixed Source Problem'
            #print phi
            
            ep = np.linalg.norm(phi-P) / np.linalg.norm(phi)
            print 'i = {}, error = {}'.format(trunc, ep)
            eps.append(ep)
        np.savetxt('UO2-{}.dat'.format(f), eps)
        
    bType = 'dlp'
    eps = []
    x = range(50)
    for trunc in x:
        P = fixedSourceSolver(sig_t, sig_s, S, lamb=0.05, truncate=trunc, bType=bType, bPattern=pat, division=div).solve(silent=True)
        
        print 'Fixed Source Problem'
        #print phi
        
        ep = np.linalg.norm(phi-P) / np.linalg.norm(phi)
        print 'i = {}, error = {}'.format(trunc, ep)
        eps.append(ep)
    np.savetxt('UO2-dlp.dat', eps)
    plotFracStudy()
    
def plotFracStudy():
    x = range(50)
    
    plt.semilogy(x, np.loadtxt('UO2-dlp.dat'), label='DLP')
    for f in np.linspace(0,1,21):
        plt.semilogy(x, np.loadtxt('UO2-{}.dat'.format(f)), label='mDLP {}'.format(f))
    
    plt.legend(loc=0,ncol=3)
    plt.ylabel('Relative error in $L_2$ norm')
    plt.xlabel('Degrees of freedom in basis')
    plt.savefig('UO2_comparison.pdf')
    plt.show()
    
def runThings():
    name = 'Triga'
    pat = np.loadtxt('phi{}.dat'.format(name))
    
    G = 238
    bType = 'mdlp'
    sig_t, sig_s, S, div = getData(G)
    
    M = np.diag(sig_t) - sig_s
    print 'Fixed Source Problem'
    phi = np.linalg.solve(M, S)
    print phi
    plotPhi(phi)
    
    eps = []
    L = div[0] if div is not None else G-1
    x = range(L)
    for trunc in x:
        P = fixedSourceSolver(sig_t, sig_s, S, lamb=0.05, truncate=trunc, bType=bType, bPattern=pat, division=div).solve(silent=True)
        
        print 'Fixed Source Problem'
        #print phi
        
        ep = np.linalg.norm(phi-P) / np.linalg.norm(phi)
        print 'i = {}, error = {}'.format(trunc, ep)
        eps.append(ep)
        
    plt.semilogy(x, eps, 'bo-')
    if div is not None:
        plt.savefig('fixedDiv{}_{}{}TRIGA.pdf'.format(len(div), bType, G))
        plt.show()
        np.savetxt('fixedDiv{}_{}{}TRIGA.dat'.format(len(div), bType, G), eps)
    else:
        plt.savefig('fixed_{}{}.pdf'.format(bType, G))
        plt.show()
        np.savetxt('fixed_{}{}.dat'.format(bType, G), eps)
        
if __name__ == "__main__":
    plotFracStudy()
    #runFracUO2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
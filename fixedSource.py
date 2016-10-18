from __future__ import division
import numpy as np
import numpy.linalg as LA
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

class fixedSourceSolver(object):
    def __init__(self, sig_t, sig_s, Q, phi=None, lamb=1, truncate=None, bType='dlp'):
        self.T = np.diag(sig_t)
        self.Q = Q
        self.S = np.array(sig_s)
        self.G = len(sig_t)
        self.phi = np.ones(self.G) if phi is None else phi
        self.lamb = lamb
        B = getBasis(bType, self.G)
        trunc = -truncate if truncate is not None else truncate
        self.basis = B.basis[:trunc]
        self.coef = B.coef[:trunc]
        
    def calcDGMCrossSections(self):
        self.Pa = self.basis.dot(self.phi)
        self.Ta = self.basis[0].dot(self.T).dot(self.phi) / self.Pa[0]
        self.da = self.basis.dot(self.T-np.eye(len(self.phi)) * self.Ta).dot(self.phi) / self.Pa[0]
        self.Sa = self.basis.dot(self.S.dot(self.phi)) / self.Pa[0]
        self.Qa = self.basis.dot(self.Q)
    
    def update(self):
        P = ((self.Sa - self.da) * self.Pa[0] + self.Qa) / self.Ta
        self.phi = (1 - self.lamb) * self.phi + self.lamb * self.basis.T.dot(P)
        
    def solve(self, analytic=None, silent=False):
        ep = 1
        i = 0
        if not silent:
            print i, ep#, self.phi
        while ep > 1e-8:
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
        return self.phi
    
class getBasis(object):
    def __init__(self, btype, G):
        if btype.lower() == 'dlp':
            self.DLP(G)
            
    def DLP(self, G):
        basis = np.zeros((G, G))
        for i in range(G):
            basis[i] = np.polynomial.Legendre.basis(i)(np.linspace(-1,1,G))
        self.basis = np.linalg.qr(basis.T, 'full')[0].T
        self.coef = np.diag(basis.T.dot(basis))
        
def getData(groups):
    if groups == 2:
        sig_t = np.array([1,2])
        sig_s = np.array([[0.3, 0],
                          [0.3, 0.3]])
        S = np.array([1, 0.000000000])
    
    
    elif groups == 3:
        sig_t = np.array([0.2822058997,0.4997685502,0.4323754911])
        sig_s = np.array([[0.2760152893, 0.0000000000, 0.0000000000],
                          [0.0011230014, 0.4533430274, 0.0000378305],
                          [0.0000000000, 0.0014582502, 0.2823864370]])
        S = np.array([0.9996892490, 0.0003391680, 0.000000000])
    
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
    
    elif groups == 238:
        # 238 group data from c5g7 geometry
        # Using UO2 data
        with h5py.File('238group.h5', 'r') as f:
            ff = f['material']['material0']
            sig_t = ff['sigma_t'][...]
            sig_s = ff['sigma_s'][...]
            S = ff['chi'][...]
    else:
        raise ValueError, "Group not yet implemented"
            
    return sig_t, sig_s, S
    
if __name__ == "__main__":
    
    G = 238
    sig_t, sig_s, S = getData(G)
    
    M = np.diag(sig_t) - sig_s
    print 'Fixed Source Problem'
    phi = np.linalg.solve(M, S)
    print phi
    
    eps = []
    for i in range(G-1):
        trunc = i if i > 0 else None
        P = fixedSourceSolver(sig_t, sig_s, S, lamb=0.05, truncate=trunc).solve(silent=True)
        
        print 'Fixed Source Problem'
        #print phi
        
        ep = np.linalg.norm(phi-P) / np.linalg.norm(phi)
        print 'i = {}, error = {}'.format(i, ep)
        eps.append(ep)
        
    plt.semilogy(range(G-1, 0, -1), eps)
    plt.savefig('fixed_{}{}.pdf'.format('dlp', G))
    plt.show()
    np.savetxt('fixed_dlp{}.dat'.format(G), eps)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
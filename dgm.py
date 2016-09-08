from __future__ import division
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
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
from numpy.polynomial import Chebyshev

class solver(object):
    def __init__(self, sig_t, sig_s, vsig_f, chi, phi, k=1, lamb=1, basis='DLP', silent=True):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.vsig_f = vsig_f
        self.chi = chi
        self.phi = phi
        self.k = k
        self.lamb = lamb
        self.ep = 1e-8
        self.basisType = basis
        self.silent = silent
        
        # Number of groups
        self.G = len(self.sig_t)
        
        self.getBasis()
        self.calcCrossSections()
#         self.solve()
        
    def calcCrossSections(self):
        # Get total cross section
        self.Sig_t = np.sum(self.basis[0] * self.sig_t * self.phi) / np.sum(self.basis[0] * self.phi) 
        # Assume isotropic flux, i.e. psi = phi / (4\pi)
        self.delta = self.basis.dot((self.sig_t - self.Sig_t) * self.phi) / np.sum(self.phi)
        
        # Get scattering cross section
        self.Sig_s = np.zeros(self.G)
        for i in range(self.G):
            for g in range(self.G):
                self.Sig_s[i] += np.sum(self.basis[i][g] * self.sig_s[:,g] * self.phi)
        self.Sig_s /= np.sum(self.phi) 
        
        # Get fission cross section
        self.vSig_f = self.basis.dot(self.vsig_f * self.phi) / np.sum(self.phi)
        self.Chi = self.basis.dot(self.chi) 

        
    def getBasis(self):
        if self.basisType == 'DLP':
            self.DLP()
        elif self.basisType == 'mDLP':
            self.mDLP()
        elif self.basisType == 'cheb':
            self.cheb()
        elif self.basisType == 'dct':
            self.DCT()
            
    def DCT(self):
        self.basis = np.zeros((self.G, self.G))
        for i in range(self.G):
            for j in range(self.G):
                self.basis[j,i] = np.cos(np.pi / self.G * (i + 0.5) * j)
    
        
    def DLP(self):
        # initialize all functions to unity 
        self.basis = np.ones((self.G, self.G))
        # Compute the linear basis function
        self.basis[:,1] = [(self.G - 1 - (2 * j)) / (self.G - 1) for j in range(self.G)]
        
        # Compute higher basis functions if needed
        if not self.G == 2:
            # Use Gram Schmidt to find the remaining basis functions
            for i in range(2, self.G):
                for j in range(self.G):
                    C0 = (i - 1) * (self.G - 1 + i)
                    C1 = (2 * i - 1) * (self.G - 1 - 2 * j)
                    C2 = i * (self.G - i)
                    self.basis[j,i] = (C1 * self.basis[j,i - 1] - C0 * self.basis[j,i - 2]) / C2
                
#         # Orthogonalize the basis functions
#         self.basis, _ = LA.qr(self.basis, 'full')
#         # Structure so that self.basis[1] provides a vector of the linear function
#         self.basis = self.basis.T

    def mDLP(self):
        self.DLP()
        pattern = np.array([1, 1])
        for i in range(len(self.basis)):
            self.basis[i] *= pattern
            
#         # Orthogonalize the basis functions
#         self.basis, _ = LA.qr(self.basis, 'full')
#         # Structure so that self.basis[1] provides a vector of the linear function
#         self.basis = self.basis.T
        
        
    def update(self):
        # Update k
        dk = self.Chi[0] * np.sum(self.vSig_f[0]) / (self.Sig_t + self.delta[0] - self.Sig_s[0])
        self.k = (1 - self.lamb) * self.k + self.lamb * dk
        # normalize the flux to phi_0 = 1
        phi = np.ones(self.G)
        # Compute higher order phi
        for i in range(1, self.G):
            dp = (self.Sig_s[i] + self.Chi[i] / self.k * np.sum(self.vSig_f[0]) - self.delta[i]) / self.Sig_t
            phi[i] = (1 - self.lamb) * phi[i] + self.lamb * dp
            
        self.phi = self.basis.dot(phi)
        self.phi /= self.phi[0]
        
    def solve(self):
        it = 0
        ep = 1
        if self.G == 2:
            while True:
                if not self.silent:
                    print 'iteration = {:3}, f = {:12.10f}, k = {:12.10f}, eps = {:12.10f}'.format(it, self.phi[1], self.k, ep)
                old = self.phi[1]
                self.calcCrossSections()
                self.update()
                it += 1
                if ep < self.ep: 
                    break
                else:
                    ep = abs(old - self.phi[1])
                
    def output(self):
        print self.Sig_t
        print self.delta
        print self.Sig_s
        print self.vSig_f

class twoGroupSolver(object):
    def __init__(self, sig_t, sig_s, vsig_f, f=1, k=1, lamb=1):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.vsig_f = vsig_f
        self.f = f
        self.k = k
        self.lamb = lamb
        self.ep = 1e-1
        
        self.getBasis()
        self.calcCrossSections()
        
    def calcCrossSections(self):
        self.Sig_t_0 = (self.sig_t[0] + self.sig_t[1] * self.f) / (1 + self.f)
        self.del_1 = (self.sig_t[0] - self.Sig_t_0 - (self.sig_t[1] - self.Sig_t_0) * self.f) / (1 + self.f)
        self.Sig_s_0 = (self.sig_s[0,0] + self.sig_s[0,1] + self.sig_s[1,1] * self.f) / (1 + self.f)
        # This is a change from eq 32 in gibson2013
        self.Sig_s_1 = (self.sig_s[0,0] - self.sig_s[0,1] - self.sig_s[1,1] * self.f) / (1 + self.f)
        self.vSig_f = (self.vsig_f[0] + self.vsig_f[1] * self.f) / (1 + self.f)
        
    def getBasis(self):
        self.basis = np.array([[1,1],[1,-1]])
        
    def getPhi(self):
        self.Phi = np.array([1 + self.f, 1 - self.f])
        
    def updatef(self):
        self.k = (1 - self.lamb) * self.k + self.lamb * self.vSig_f / (self.Sig_t_0 - self.Sig_s_0)
        phi1_0 = (self.Sig_s_1 + self.vSig_f / self.k - self.del_1) / self.Sig_t_0
        self.f = (1 - self.lamb) * self.f + self.lamb * (1 - phi1_0) / (1 + phi1_0)
        
    def solve(self):
        it = 0
        ep = 1
        while True:
            print 'iteration = {:3}, f = {:12.10f}, k = {:12.10f}, eps = {:12.10f}'.format(it, self.f, self.k, ep)
            oldf = self.f
            self.calcCrossSections()
            self.updatef()
            it += 1
            if ep < self.ep: 
                break
            else:
                ep = abs(oldf - self.f)
                
    def output(self):
        print self.Sig_t_0
        print self.del_1
        print self.Sig_s_0, self.Sig_s_1
        print self.vSig_f
        
if __name__ == '__main__':
    N = 50
    sig_t = np.array([1,3])
    sig_s = np.array([[0.3, 0.3],
                      [0, 0.3]])
    vsig_f = np.array([0.5, 0.5])
    chi = np.array([1,0])
    phi = np.array([1,0.05])
    input = np.linspace(0,0.5,N)
    fs = np.zeros(N)
    for i, f in enumerate(input):
        phi = np.array([1, f])
        S = solver(sig_t, sig_s, vsig_f, chi, phi, basis='DLP')
        S.update()
        fs[i] = S.phi[1] / S.phi[0]
        print S.phi[1] / S.phi[0]
    
    plt.plot(input, fs, 'b-')
    plt.plot(input, input, 'k--')
    plt.plot((input[1:] + input[:-1]) / 2.0, (fs[1:] - fs[:-1]) / (input[1:] - input[:-1]), 'r--')
    plt.ylim(-1.5, 1.0)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend(['f(x)', 'y=x', 'f\'(x)'], loc=4)
    plt.xticks(np.linspace(0,0.5,11))
    plt.grid()
    plt.savefig('dlp3.pdf')
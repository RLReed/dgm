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

class solver(object):
    def __init__(self, sig_t, sig_s, vsig_f, chi, phi=None, k=1, lamb=1, basis='dlp', silent=True, pattern=None):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.vsig_f = vsig_f
        self.chi = chi
        if phi is None:
            self.phi = np.ones(len(sig_t))
        else:
            self.phi = phi
        self.pattern = pattern
        self.k = k
        self.lamb = lamb
        self.ep = 1e-8
        self.basisType = basis
        self.silent = silent
        
        # Number of groups
        self.G = len(self.sig_t)
        
        self.getBasis()
        self.calcCrossSections()
        
    def calcCrossSections(self):
        # Get total cross section
        self.Sig_t = self.sig_t.dot(self.phi) / self.basis[0].dot(self.phi)

        self.delta = self.basis.dot(np.diag(self.sig_t - self.Sig_t)).dot(self.phi) / self.basis[0].dot(self.phi)
        self.delta[0] = 0
        
        # Get scattering cross section
        self.Sig_s = self.basis.dot(self.sig_s).dot(self.phi) / self.basis[0].dot(self.phi)
        
        # Get fission cross section
        self.vSig_f = self.vsig_f.dot(self.phi) / self.basis[0].dot(self.phi)
        self.Chi = self.basis.dot(self.chi) 
        
    def getBasis(self):
        if self.basisType.lower() == 'dlp':
            self.DLP()
        elif self.basisType.lower() == 'mdlp':
            self.mDLP()
        elif self.basisType.lower() == 'cheb':
            self.cheb()
        elif self.basisType.lower() == 'dct':
            self.DCT()
        elif self.basisType.lower() == 'mod':
            self.mod()
        else:
            raise ValueError('Basis not implemented')
        
        self.basis /= self.basis[0,0]
        if not self.silent:
            for g in range(self.G):
                print ('basis_{} = [' + (' {: .8f}' * self.G)[1:] + ']').format(g, *self.basis[g])
            print # blank line
         
    def mod(self):
        assert len(self.pattern) == self.G
        self.basis = np.ones((self.G, self.G))
        self.basis[1] = np.array([1.00000000, 0.02423474, 0.00023562])
        self.basis[2] = np.array([1.00000000, 0.02423474, 0.00023562])
        
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis.T, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        self.basis = self.basis.T
        
    def DCT(self):
        self.basis = np.zeros((self.G, self.G))
        for i in range(self.G):
            for j in range(self.G):
                self.basis[j,i] = np.cos(np.pi / self.G * (i + 0.5) * j)
        
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis, 'full')
       
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
                
                
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        self.basis = self.basis.T

    def mDLP(self):
        self.DLP()
        pattern = np.array([1.00000000, 0.02423474, 0.00023562])
        for i in range(len(self.basis)):
            self.basis[i] *= pattern
            
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        self.basis = self.basis.T
        
    def update(self):
        # Expand phi
        Phi = self.basis.dot(self.phi)
        print Phi
        
        # Find phi
        Phi = ((self.Sig_s + self.Chi * self.vSig_f / self.k - self.delta) / self.Sig_t) * Phi[0]
        
        # Update k
        self.k = (self.Chi[0] * self.vSig_f) / (self.Sig_t - self.Sig_s[0])
        
        self.phi = self.basis.T.dot(Phi)
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
        else:
            self.ks = []
            self.phis = []
            while True:
                self.ks.append(self.k)
                self.phis.append(self.phi.copy())
                if not self.silent:
                    print 'iteration = {:3}, k = {:12.10f}, eps = {:12.10f}'.format(it, self.k, ep)
                    print 'phi = {}'.format(self.phi)
                old = self.phi.copy()
                self.calcCrossSections()
                self.update()
                it += 1
                if ep < self.ep: 
                    break
                else:
                    ep = LA.norm(self.phi - old)
            self.phis = np.array(self.phis)
            self.makePlots(it)
                
    def makePlots(self, it):
        plt.plot(range(it), self.ks)
        plt.xlabel('iterations')
        plt.ylabel('k-eigenvalue')
        plt.grid()
        plt.savefig('k_v_iteration_{}.pdf'.format(basis))
        plt.clf()
        
        plt.semilogy(range(it), self.ks, 'r:', label='k-eigenvalue')
        plt.semilogy(range(it), self.phis[:,1], 'b-', label='$\phi_1$')
        plt.semilogy(range(it), self.phis[:,2], 'g--', label='$\phi_2$')
        plt.xlabel('iterations')
        plt.ylabel('$\phi$ or k-eigenvalue')
        plt.grid()
        plt.legend(ncol=self.G)
        plt.title('Iteration results using {} basis'.format(self.basisType))
        plt.savefig('phi_v_iteration_{}.pdf'.format(basis))
        plt.clf()
                
    def output(self):
        print 'Sig_t = {}'.format(self.Sig_t)
        print 'delta = {}'.format(self.delta)
        print 'Sig_t = {}'.format(self.Sig_s)
        print 'vSig_f= {}'.format(self.vSig_f)
        print 'chi   = {}'.format(self.chi)

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
    basis = 'mod'
    
    # 3-group data from roberts' thesis
    sig_t = np.array([0.2822058997,0.4997685502,0.4323754911])
    sig_s = np.array([[0.2760152893, 0.0000000000, 0.0000000000],
                      [0.0011230014, 0.4533430274, 0.0000378305],
                      [0.0000000000, 0.0014582502, 0.2823864370]])
    v = np.array([2.7202775245, 2.4338148428, 2.4338000000])
    sig_f = np.array([0.0028231045, 0.0096261203, 0.1123513981])
    vsig_f = v * sig_f
    chi = np.array([0.9996892490, 0.0003391680, 0.000000000])
    
    # Solve analytic:
    k = 1
    M = np.diag(sig_t) - sig_s
    M = np.linalg.inv(M).dot(np.outer(chi,vsig_f))
    eigs = np.linalg.eig(M)
    print 'Analytic Solution'
    print 'k = {}'.format(eigs[0][0])
    print 'phi = {:.8f} {:.8f} {:.8f}'.format(*[i for i in eigs[1][:,0]])
    print 'normed phi = {:.8f} {:.8f} {:.8f}'.format(*[i / eigs[1][0,0] for i in eigs[1][:,0]])
    analytic = np.array([i / eigs[1][0,0] for i in eigs[1][:,0]])
    print # blank line
    phi = np.ones(3)
    ep = 1
    phis = []
    it = 0
    while True:
        it += 1
        old = phi.copy()
        phi = M.dot(phi)
        phi /= LA.norm(phi)
        print phi
        ep = LA.norm(phi - old)
        phis.append(phi.copy())
        if ep < 1e-8: break
        
    phis = np.array(phis)
    print it, phis[0], (M.dot(phi) / phi)[0]
    plt.semilogy(range(it), phis)
    plt.show()
    asdf
        
    
    
    inputs = np.linspace(0,0.5,N)
    fs = np.zeros(N)
    S = solver(sig_t, sig_s, vsig_f, chi, basis=basis, silent=False, pattern=analytic)
    S.solve()
    asdf
    for i, f in enumerate(inputs):
        phi = np.array([1, f, f ** 2])
        S = solver(sig_t, sig_s, vsig_f, chi, phi=phi, basis=basis, pattern=analytic)
#         S.output()
        S.update()
        fs[i] = S.phi[1] / S.phi[0]
        print S.phi[1] / S.phi[0]
    
    plt.plot(inputs, fs, 'b-')
    plt.plot(inputs, inputs, 'k--')
    plt.plot((inputs[1:] + inputs[:-1]) / 2.0, (fs[1:] - fs[:-1]) / (inputs[1:] - inputs[:-1]), 'r--')
    plt.ylim(-1.5, 1.0)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend(['f(x)', 'y=x', 'f\'(x)'], loc=4)
    plt.xticks(np.linspace(0,0.5,11))
    plt.grid()
    plt.savefig('{}{}.pdf'.format(basis, sig_t[1]))
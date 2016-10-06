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
import h5py

class solver(object):
    def __init__(self, sig_t, sig_s, vsig_f, chi, phi=None, k=1, lamb=1, basis='dlp', silent=True, pattern=None, division=None, analyticError=False):
        # Store all arguments
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
        self.analyticError = analyticError
        
        # Number of fine groups
        self.nFine = len(self.sig_t)
        
        # Determine course group structure
        self.courseStruct = [0]
        if division is not None:
            assert max(division) < self.nFine, "Group division outside of available groups"
            self.courseStruct = [0] + division
            course = np.array(self.courseStruct + [self.nFine])
            self.fineStruct = course[1:] - course[:-1]
        else:
            self.fineStruct = [self.nFine]
            
        # Number of course groups
        self.nCourse = len(self.courseStruct)
        
        if basis != 'mod':
            self.basis = [self.getBasis(G) for G in self.fineStruct]
        else:
            self.b = [0] + np.cumsum(self.fineStruct).tolist()
            pat = [pattern[self.b[i]:self.b[i+1]] for i in range(self.nCourse)]
            
            self.basis = [self.getBasis(G, pat[i]) for G in self.fineStruct]
        
    def calcCrossSections(self):
        self.splitPhi()
        
        # Initialize cross sections
        self.Sig_t = np.zeros(self.nCourse).tolist()
        self.delta = np.zeros(self.nCourse).tolist()
        self.Sig_s = np.zeros((self.nCourse, self.nCourse)).tolist()
        self.vSig_f = np.zeros(self.nCourse).tolist()
        self.Chi = np.zeros(self.nCourse).tolist()

        # Loop over course group to colapse the fine groups
        for G in range(self.nCourse):
            phiG = self.basis[G][0].dot(self.phi[G]) if len(self.basis[G]) > 1 else self.phi[G]
            # Get total cross section
            sig_t = self.sig_t[self.b[G]:self.b[G+1]]
            self.Sig_t[G] = sig_t.dot(self.phi[G]) / phiG

            self.delta[G] = self.basis[G].dot(np.diag(sig_t - self.Sig_t[G])).dot(self.phi[G]) / phiG
            self.delta[G][0] = 0
            
            # Get fission cross section
            self.vSig_f[G] = self.vsig_f[self.b[G]:self.b[G+1]].dot(self.phi[G]) / phiG
            self.Chi[G] = self.basis[G].dot(self.chi[self.b[G]:self.b[G+1]]) 
            
            # Get scattering cross section
            for Gp in range(self.nCourse):   
                sig_s = self.sig_s[self.b[G]:self.b[G+1],self.b[Gp]:self.b[Gp+1]]
                phiGp = self.basis[Gp][0].dot(self.phi[Gp]) if len(self.basis[Gp]) > 1 else self.phi[Gp]
                self.Sig_s[G][Gp] = self.basis[G].dot(sig_s).dot(self.phi[Gp]) / phiGp
        
        self.joinPhi()
        #self.output()
        #print 
        
    def getBasis(self, G, pattern=None):
        if G > 1:
            if self.basisType.lower() == 'dlp':
                basis = self.DLP(G)
            elif self.basisType.lower() == 'mdlp':
                basis = self.mDLP(G)
            elif self.basisType.lower() == 'cheb':
                basis = self.cheb(G)
            elif self.basisType.lower() == 'dct':
                basis = self.DCT(G)
            elif self.basisType.lower() == 'mod':
                basis = self.mod(G, pattern)
            else:
                raise ValueError('Basis not implemented')
            
            basis /= basis[0,0]
            if False:
                for g in range(G):
                    print ('basis_{} = [' + (' {: .8f}' * G)[1:] + ']').format(g, *basis[g])
                print # blank line
        else:
            basis = np.ones(1)
        return basis
         
    def mod(self, G, pattern):
        basis = np.ones((G, G))
        basis[1:] = np.array(pattern)
        
        # Orthogonalize the basis functions
        basis, _ = LA.qr(basis.T, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        return basis.T
        
    def DCT(self):
        self.basis = np.zeros((self.G, self.G))
        for i in range(self.G):
            for j in range(self.G):
                self.basis[j,i] = np.cos(np.pi / self.G * (i + 0.5) * j)
        
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis, 'full')
       
    def DLP(self, G):
        # initialize all functions to unity 
        basis = np.ones((G, G))
        # Compute the linear basis function
        basis[:,1] = [(G - 1 - (2 * j)) / (G - 1) for j in range(G)]
        
        # Compute higher basis functions if needed
        if not G == 2:
            # Use Gram Schmidt to find the remaining basis functions
            for i in range(2, G):
                for j in range(G):
                    C0 = (i - 1) * (G - 1 + i)
                    C1 = (2 * i - 1) * (G - 1 - 2 * j)
                    C2 = i * (G - i)
                    basis[j,i] = (C1 * basis[j,i - 1] - C0 * basis[j,i - 2]) / C2
                
                
        # Orthogonalize the basis functions
        basis, _ = LA.qr(basis, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        return basis.T
        
    def mDLP(self):
        self.DLP()
        pattern = np.array([1.00000000, 0.02423474, 0.00023562])
        for i in range(len(self.basis)):
            self.basis[i] *= pattern
            
        # Orthogonalize the basis functions
        self.basis, _ = LA.qr(self.basis, 'full')
        # Structure so that self.basis[1] provides a vector of the linear function
        self.basis = self.basis.T
         
    def splitPhi(self):
        # split phi along course group divisions
        self.b = [0] + np.cumsum(self.fineStruct).tolist()
        self.phi = [self.phi[self.b[i]:self.b[i+1]] for i in range(self.nCourse)]
        
    def joinPhi(self):     
        self.phi = [phi.tolist() for phi in self.phi]
        self.phi = [[phi] if type(phi) == float else phi for phi in self.phi]
        self.phi = sum(self.phi, [])
        #self.phi = np.array([max(p,0) for p in self.phi])
        self.phi = np.array(self.phi)
         
    def update(self):
        self.splitPhi()
        
        # Expand phi
        Phi = [self.basis[G].dot(self.phi[G]) for G in range(self.nCourse)]
        phi0 = np.array([p[0] if type(p) != np.float64 else p for p in Phi])
#         if type(Phi[0]) != np.float64:
#             if type(Phi[1]) != np.float64:
#                 self.coefRatio = Phi[0][0] / Phi[0][1]
#             else:
#                 self.coefRatio = Phi[0][0] / Phi[0][1]
        
        for G in range(self.nCourse):
            s = np.array(self.Sig_s[G]).T.dot(phi0)
            f = np.array(self.Chi[G]) / self.k * phi0.dot(self.vSig_f)
            d = np.array(self.delta[G]).dot(Phi[G][0]) if type(Phi[G]) != np.float64 else self.delta[G] * Phi[G]
            Phi[G] = (s + f - d) / self.Sig_t[G]
             
        for G in range(self.nCourse):
            self.phi[G] = (1 - self.lamb) * self.phi[G] + self.lamb * self.basis[G].T.dot(Phi[G])
            
        self.joinPhi()
        self.phi /= self.phi[0]
        
        # Update k        
        
        t = np.sum([self.sig_t[g] * self.phi[g] for g in range(self.nFine)])
        s = np.sum([np.array(self.sig_s[g]).dot(self.phi) for g in range(self.nFine)])
        f = np.sum([self.chi[g] * np.array(self.vsig_f).dot(self.phi) for g in range(self.nFine)])
        
        self.k = (1 - self.lamb) * self.k + self.lamb * f / (t - s)
        
    def solve(self):
        it = 0
        ep = 1
        if self.nFine == 2 and self.nCourse == 1:
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
            self.coefs = []
            while True:
                old = self.phi[:]
                self.ks.append(self.k)
                self.phis.append(self.phi[:])
                if not self.silent:
                    print 'iteration = {:3}, k = {:12.10f}, eps = {:12.10f}'.format(it, self.k, ep)
                    print 'phi = {}'.format(self.phi)
                self.calcCrossSections()
                self.update()
#                 self.coefs.append(self.coefRatio)
                it += 1
                #if it < 100:
                #self.phi = np.array([p if p > 0 else 0.5 * self.phis[-1][i] for i, p in enumerate(self.phi)])
                if ep < self.ep: 
                    break
                else:
                    if self.analyticError:
                        ep = LA.norm(self.phi - self.pattern) / LA.norm(self.pattern)
                    else:
                        ep = LA.norm(self.phi - old) / LA.norm(old)
            print 'analytic error = {}'.format(LA.norm(self.phi - self.pattern) / LA.norm(self.pattern))
            self.phis = np.array(self.phis)
            self.makePlots(it)
                
    def makePlots(self, it):
        plt.plot(range(it), self.ks)
        plt.xlabel('iterations')
        plt.ylabel('k-eigenvalue')
        plt.grid()
        plt.savefig('k_v_iteration_{}.pdf'.format(self.basisType))
        plt.clf()
        
        plt.semilogy(range(it), self.ks, 'r:', label='k-eigenvalue')
        plt.semilogy(range(it), self.phis[:,1], 'b-', label='$\phi_1$')
        plt.semilogy(range(it), self.phis[:,2], 'g--', label='$\phi_2$')
        plt.xlabel('iterations')
        plt.ylabel('$\phi$ or k-eigenvalue')
        plt.grid()
        plt.legend(ncol=self.nCourse)
        plt.title('Iteration results using {} basis'.format(self.basisType))
        plt.savefig('phi_v_iteration_{}.pdf'.format(self.basisType))
        plt.clf()
        
#         plt.plot(range(it), self.coefs)
#         plt.xlabel('iterations')
#         plt.ylabel('$a_0 / a_1$')
#         plt.grid()
#         plt.savefig('coefRatio.pdf'.format(self.basisType))
#         plt.clf()
                
    def output(self):
        print 'Sig_t = {}'.format(self.Sig_t)
        print 'delta = {}'.format(self.delta)
        print 'Sig_s = {}'.format(self.Sig_s)
        print 'vSig_f= {}'.format(self.vSig_f)
        print 'Chi   = {}'.format(self.Chi)

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
    basis = 'dlp'
    groups = 238
    
    np.set_printoptions(suppress=True)
    if groups == 3:
        # 3-group data from roberts' thesis
        sig_t = np.array([0.2822058997,0.4997685502,0.4323754911])
        sig_s = np.array([[0.2760152893, 0.0000000000, 0.0000000000],
                          [0.0011230014, 0.4533430274, 0.0000378305],
                          [0.0000000000, 0.0014582502, 0.2823864370]])
        v = np.array([2.7202775245, 2.4338148428, 2.4338000000])
        sig_f = np.array([0.0028231045, 0.0096261203, 0.1123513981])
        chi = np.array([0.9996892490, 0.0003391680, 0.000000000])
        division=None
        division=[1,2]
    elif groups == 7:
        sig_t = np.array([0.21245,0.355470, 0.48554, 0.5594, 0.31803, 0.40146, 0.57061])
        sig_s = np.array([[1.27537e-1, 4.2378e-2, 9.4374e-6, 5.5163e-9, 0, 0, 0],
                          [0, 3.24456e-1, 1.6314e-3, 3.1427e-9, 0,0,0],
                          [0, 0, 4.5094e-1, 2.6792e-3, 0, 0, 0],
                          [0, 0, 0, 4.52565e-1, 5.5664e-3, 0, 0],
                          [0, 0, 0, 1.2525e-4, 2.71401e-1, 1.0255e-2, 1.0021e-8],
                          [0, 0, 0, 0, 1.2968e-3, 2.65802e-1, 1.6809e-2],
                          [0, 0, 0, 0, 0, 8.5458e-3, 2.7308e-1]]).T
        v = np.array([2.78145, 2.47443, 2.43383, 2.43380, 2.4438, 2.4338, 2.4338])
        sig_f = np.array([7.21206e-3, 8.19301e-4, 6.4532e-3, 1.85648e-2, 1.78084e-2, 8.30348e-2, 2.16004e-1])
        chi = np.array([5.87910e-1, 4.1176e-1, 3.3906e-4, 1.1761e-7, 0, 0, 0])
        division = [4]
        division = range(1,7)
        #division = None
    elif groups == 238:
        # 238 group data from c5g7 geometry
        # Using UO2 data
        with h5py.File('238group.h5', 'r') as f:
            ff = f['material']['material0']
            sig_t = ff['sigma_t'][...]
            sig_s = ff['sigma_s'][...]
            v = ff['nu'][...]
            sig_f = ff['sigma_f'][...]
            chi = ff['chi'][...]
            
        e25 = [ 2.00000000e+07,
               1.00000000e+07,
               3.00000000e+06,
               1.35600000e+06,
               4.99520000e+05,9.11800000e+03,1.48729000e+02,4.80520000e+01,   1.59680000e+01,   9.87700000e+00,4.00000000e+00,   1.30000000e+00,   1.09700000e+00,1.02000000e+00,   9.50000000e-01,   6.25000000e-01,3.50000000e-01,   2.80000000e-01,   1.80000000e-01,1.40000000e-01,   1.00000000e-01,   8.00000000e-02,5.80000000e-02,   4.20000000e-02,   3.00000000e-02,1.50000000e-02,   1.00000000e-05]
        
        e238 = [2.00000000e+07,1.73330000e+07,1.56830000e+07,1.45500000e+07,1.38400000e+07,1.28400000e+07,
                1.00000000e+07,8.18730000e+06,6.43400000e+06,4.80000000e+06,4.30400000e+06,3.00000000e+06,
                2.47900000e+06,2.35400000e+06,1.85000000e+06,1.50000000e+06,1.40000000e+06,1.35600000e+06,
                1.31700000e+06,1.25000000e+06,1.20000000e+06,1.10000000e+06,1.01000000e+06,9.20000000e+05,9.00000000e+05,8.75000000e+05,8.61100000e+05,8.20000000e+05,7.50000000e+05,6.79000000e+05,6.70000000e+05,6.00000000e+05,5.73000000e+05,5.50000000e+05,
                4.99520000e+05,   4.70000000e+05,4.40000000e+05,   4.20000000e+05,   4.00000000e+05,3.30000000e+05,   2.70000000e+05,   2.00000000e+05,1.50000000e+05,   1.28300000e+05,   1.00000000e+05,8.50000000e+04,   8.20000000e+04,   7.50000000e+04,7.30000000e+04,   6.00000000e+04,   5.20000000e+04,5.00000000e+04,   4.50000000e+04,   3.00000000e+04,2.50000000e+04,   1.70000000e+04,   1.30000000e+04,9.50000000e+03,   8.03000000e+03,   6.00000000e+03,3.90000000e+03,   3.74000000e+03,   3.00000000e+03,2.58000000e+03,   2.29000000e+03,   2.20000000e+03,1.80000000e+03,   1.55000000e+03,   1.50000000e+03,1.15000000e+03,   9.50000000e+02,   6.83000000e+02,6.70000000e+02,   5.50000000e+02,   3.05000000e+02,2.85000000e+02,   2.40000000e+02,   2.10000000e+02,2.07500000e+02,   1.92500000e+02,   1.86000000e+02,1.22000000e+02,   1.19000000e+02,   1.15000000e+02,1.08000000e+02,   1.00000000e+02,   9.00000000e+01,8.20000000e+01,   8.00000000e+01,   7.60000000e+01,7.20000000e+01,   6.75000000e+01,   6.50000000e+01,6.10000000e+01,   5.90000000e+01,   5.34000000e+01,5.20000000e+01,   5.06000000e+01,   4.92000000e+01,4.83000000e+01,   4.70000000e+01,   4.52000000e+01,4.40000000e+01,   4.24000000e+01,   4.10000000e+01,3.96000000e+01,   3.91000000e+01,   3.80000000e+01,3.70000000e+01,   3.55000000e+01,   3.46000000e+01,3.37500000e+01,   3.32500000e+01,   3.17500000e+01,3.12500000e+01,   3.00000000e+01,   2.75000000e+01,2.50000000e+01,   2.25000000e+01,   2.10000000e+01,2.00000000e+01,   1.90000000e+01,   1.85000000e+01,1.70000000e+01,   1.60000000e+01,   1.51000000e+01,1.44000000e+01,   1.37500000e+01,   1.29000000e+01,1.19000000e+01,   1.15000000e+01,   1.00000000e+01,9.10000000e+00,   8.10000000e+00,   7.15000000e+00,7.00000000e+00,   6.75000000e+00,   6.50000000e+00,6.25000000e+00,   6.00000000e+00,   5.40000000e+00,5.00000000e+00,   4.75000000e+00,   4.00000000e+00,3.73000000e+00,   3.50000000e+00,   3.15000000e+00,3.05000000e+00,   3.00000000e+00,   2.97000000e+00,2.87000000e+00,   2.77000000e+00,   2.67000000e+00,2.57000000e+00,   2.47000000e+00,   2.38000000e+00,2.30000000e+00,   2.21000000e+00,   2.12000000e+00,2.00000000e+00,   1.94000000e+00,   1.86000000e+00,1.77000000e+00,   1.68000000e+00,   1.59000000e+00,1.50000000e+00,   1.45000000e+00,   1.40000000e+00,1.35000000e+00,   1.30000000e+00,   1.25000000e+00,1.22500000e+00,   1.20000000e+00,   1.17500000e+00,1.15000000e+00,   1.14000000e+00,   1.13000000e+00,1.12000000e+00,   1.11000000e+00,   1.10000000e+00,1.09000000e+00,   1.08000000e+00,   1.07000000e+00,1.06000000e+00,   1.05000000e+00,   1.04000000e+00,1.03000000e+00,   1.02000000e+00,   1.01000000e+00,1.00000000e+00,   9.75000000e-01,   9.50000000e-01,9.25000000e-01,   9.00000000e-01,   8.50000000e-01,8.00000000e-01,   7.50000000e-01,   7.00000000e-01,6.50000000e-01,   6.25000000e-01,   6.00000000e-01,5.50000000e-01,   5.00000000e-01,   4.50000000e-01,4.00000000e-01,   3.75000000e-01,   3.50000000e-01,3.25000000e-01,   3.00000000e-01,   2.75000000e-01,2.50000000e-01,   2.25000000e-01,   2.00000000e-01,1.75000000e-01,   1.50000000e-01,   1.25000000e-01,1.00000000e-01,   9.00000000e-02,   8.00000000e-02,7.00000000e-02,   6.00000000e-02,   5.00000000e-02,4.00000000e-02,   3.00000000e-02,   2.53000000e-02,1.00000000e-02,   7.50000000e-03,   5.00000000e-03,4.00000000e-03,   3.00000000e-03,   2.50000000e-03,2.00000000e-03,   1.50000000e-03,   1.20000000e-03,1.00000000e-03,   7.50000000e-04,   5.00000000e-04,1.00000000e-04,   1.00000000e-05]
#         plt.semilogy(sig_t)
#         plt.show()
        l = []
        for e in e25:
            for i, E in enumerate(e238):
                if E < e:
                    l.append(i-1)
                    break
        division = [8, 16, 30, 40, 57, 63, 75, 93, 103, 111, 116, 121, 132, 139, 160, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235]
        plt.plot(sig_t)
        plt.plot(division, np.ones(len(division)), 'go')
        #plt.show()
        #division = None
        division = range(2,236,2)
        
    vsig_f = v * sig_f
    # Solve analytic:
    k = 1
    M = np.diag(sig_t) - sig_s
    M = np.linalg.inv(M).dot(np.outer(chi,vsig_f))
    eigs = np.linalg.eig(M)
    eig = np.zeros(2).tolist()
    eig[0] = np.real(eigs[0])
    eig[1] = np.real(eigs[1])
    eigs = eig
    print 'Analytic Solution'
    print 'k = {}'.format(eigs[0][0])
    print 'phi = {:.8f} {:.8f} {:.8f}'.format(*[i for i in eigs[1][:,0]])
    print 'normed phi = {:.8f} {:.8f} {:.8f}'.format(*[i / eigs[1][0,0] for i in eigs[1][:,0]])
    print np.array([i / eigs[1][0,0] for i in eigs[1][:,0]])
    
    analytic = np.array([i / eigs[1][0,0] for i in eigs[1][:,0]])
    analK = eigs[0][0]
    print # blank line
    phi = np.ones(6)
    ep = 1
    phis = []
    it = 0
#     while True:
#         it += 1
#         old = phi.copy()
#         phi = M.dot(phi)
#         phi /= LA.norm(phi)
#         print phi
#         ep = LA.norm(phi - old)
#         phis.append(phi.copy())
#         if ep < 1e-8: break
#         
#     phis = np.array(phis)
#     print it, phis[0], (M.dot(phi) / phi)[0]
#     plt.semilogy(range(it), phis)
#     plt.show()
#     plt.clf
        
    
    
    inputs = np.linspace(0,0.5,N)
    fs = np.zeros(N)
    S = solver(sig_t, sig_s, vsig_f, chi, k=analK, phi=analytic, basis=basis, silent=False, pattern=analytic, division=division, lamb=0.3)
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
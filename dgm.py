from __future__ import division
import numpy as np

class twoGroupSolver(object):
    def __init__(self, sig_t, sig_s, vsig_f, f=1, k=1, lamb=1):
        self.sig_t = sig_t
        self.sig_s = sig_s
        self.vsig_f = vsig_f
        self.f = f
        self.k = k
        self.lamb = lamb
        self.ep = 1e-12
        
        self.getBasis()
        self.solve()
        
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
        
        
if __name__ == '__main__':
    sig_t = np.array([1,3])
    sig_s = np.array([[0.3, 0.3],
                      [0, 0.3]])
    vsig_f = np.array([0.5, 0.5])
    twoGroupSolver(sig_t, sig_s, vsig_f, lamb=0.7)

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
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
                           cycler('linestyle', ['-', '-', '-', '-', '-', '--', '--', '--', '--', '--', ':', ':', ':', ':', ':', '-.', '-.', '-.', '-.', '-.']) + 
                           cycler('marker', ['', '', '', '', 'o', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''])))
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

class fixedSourceSolver(object):
    def __init__(self, sig_t, sig_s, vsig_f, chi, phi=None, k=None, lamb=1, truncate=None, bType='dlp', bPattern=None, division=None):
        self.sig_t = np.array(sig_t)
        self.vsig_f = vsig_f
        self.chi = chi
        self.sig_s = np.array(sig_s)
        self.phi = np.ones(len(sig_t)) if phi is None else phi
        self.k = 1 if k is None else k
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
            pat = bPattern if bPattern is None else bPattern[self.bounds[G]:self.bounds[G+1]] 
            B = getBasis(bType, groups, pat)
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
        
        def getFission(vsig_f, phi, basis):
            return vsig_f.dot(phi) / basis[0].dot(phi)
        
        def getChi(chi, basis):
            return basis.dot(chi)
        
        self.Pa = []
        self.Ta = []
        self.da = []
        self.Sa = []
        self.Fa = []
        self.Xa = []
        for G in range(self.nCourse):
            # Set reused parameters
            T = self.sig_t[self.bounds[G]:self.bounds[G+1]]
            phiG = self.phi[self.bounds[G]:self.bounds[G+1]]
            # Get expanded phi
            self.Pa.append(getPhiCoef(phiG, self.basis[G]))
            # Get total DGM cross section
            self.Ta.append(getTotal(T, phiG, self.basis[G]))
            # Get delta DGM cross section
            self.da.append(getDelta(T, self.Ta[G], phiG, self.basis[G]))
            # Get fission DGM cross section
            self.Fa.append(getFission(self.vsig_f[self.bounds[G]:self.bounds[G+1]], phiG, self.basis[G]))
            # Get DGM chi spectrum
            self.Xa.append(getChi(self.chi[self.bounds[G]:self.bounds[G+1]], self.basis[G]))
            # Get Scattering DGM cross section
            Sa = []
            for Gp in range(self.nCourse):
                sig_s = self.sig_s[self.bounds[G]:self.bounds[G+1], self.bounds[Gp]:self.bounds[Gp+1]]
                phi = self.phi[self.bounds[Gp]:self.bounds[Gp+1]]
                phi0 = self.basis[Gp][0].dot(phi)
                Sa.append(getScatter(sig_s, phi, self.basis[G], phi0))
            self.Sa.append(Sa)
        #self.outputCrossSections()
        
    def update(self):
        # Loop over course groups
        for G in range(self.nCourse):
            S = 0
            F = 0
            for Gp in range(self.nCourse):
                S += self.Sa[G][Gp] * self.Pa[Gp][0]
                F += self.Fa[Gp] * self.Pa[Gp][0]
            self.Pa[G] = (S + self.Xa[G] / self.k * F - self.da[G] * self.Pa[G][0]) / self.Ta[G]
        phi = []
        for G in range(self.nCourse):
            phi.append(self.basis[G].T.dot(self.Pa[G]))
        phi = np.concatenate(phi)
        
        self.phi = (1 - self.lamb) * self.phi + self.lamb * phi
        self.k = np.sum(self.chi * self.vsig_f.dot(self.phi)) / np.sum(np.diag(self.sig_t).dot(self.phi) - self.sig_s.dot(self.phi))
        
    def solve(self, analytic=None, silent=False):
        ep = 1
        i = 0
        if not silent:
            print 'it = {}, ep = {}, k = {}\r'.format(i, ep, self.k),
        while ep > 1e-12:
            old = self.phi.copy()
            self.calcDGMCrossSections()
            self.update()
            if analytic is None:
                ep = norm(old-self.phi)
            else:
                ep = norm(analytic-self.phi)
            i += 1
            if not silent:
                print 'it = {}, ep = {}, k = {}\r'.format(i, ep, self.k),
            if i > 10000 and False:
                print 'WARNING: iteration not converged'
                break
        if not silent:
            print
        return self.phi, self.k
    
    def outputCrossSections(self):
        print 'PA = {}'.format(self.Pa)
        print 'TA = {}'.format(self.Ta)
        print 'DA = {}'.format(self.da)
        print 'SA = {}'.format(self.Sa)
        print 'FA = {}'.format(self.Fa)
        print 'XA = {}'.format(self.Xa)
    
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
        vsig_f = np.array([0.5,0.5])
        chi = np.array([1, 0.000000000])
        division = None
        #division = [1]
    
    elif groups == 3:
        sig_t = np.array([0.2822058997,0.4997685502,0.4323754911])
        sig_s = np.array([[0.2760152893, 0.0000000000, 0.0000000000],
                          [0.0011230014, 0.4533430274, 0.0000378305],
                          [0.0000000000, 0.0014582502, 0.2823864370]])
        v = np.array([2.7202775245, 2.4338148428, 2.4338000000])
        sig_f = np.array([0.0028231045, 0.0096261203, 0.1123513981])
        chi = np.array([0.9996892490, 0.0003391680, 0.000000000])
        vsig_f = v * sig_f
        division=None
        #division=[2]
    
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
        vsig_f = v * sig_f
        division = [4]
        
    elif groups == 69:
        # 69 group data from uo2 pin cell geometry
        with h5py.File('69-groupData/uo2XS-0.3-69.h5', 'r') as f:
            sig_t = f['sig_t'][...]
            sig_s = f['sig_s'][...]
            vsig_f = f['vsig_f'][...]
            chi = f['chi'][...]
        division = [27, 54]
        #division = None
    
    elif groups == 238:
        # 238 group data from c5g7 geometry
        # Using UO2 data
        with h5py.File('uo2XS-0.3.h5', 'r') as f:
            sig_t = f['sig_t'][...]
            sig_s = f['sig_s'][...]
            vsig_f = f['vsig_f'][...]
            chi = f['chi'][...]
        division = [50, 100, 150, 200]
        #division = None
    else:
        raise ValueError, "Group not yet implemented"
            
    return sig_t, sig_s, vsig_f, chi, division

def getMax(x):
    x = np.concatenate(x, [len(x)])
    x = np.concatenate([0], x)
    return x[1:] - x[:-1]

def plotPhi(phi):
    plt.clf()
    if len(phi) == 238:
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
        div = [50, 100, 150, 200]
    elif len(phi) == 69:
        E = np.array([1.00000000e+07,6.06550000e+06,3.67900000e+06,2.23100000e+06,1.35300000e+06,
                      8.21000000e+05,5.00000000e+05,3.02500000e+05,1.83000000e+05,1.11000000e+05,
                      6.73400000e+04,4.08500000e+04,2.47800000e+04,1.50300000e+04,9.11800000e+03,
                      5.53000000e+03,3.51910000e+03,2.23945000e+03,1.42510000e+03,9.06899000e+02,
                      3.67263000e+02,1.48729000e+02,7.55014000e+01,4.80520000e+01,2.77000000e+01,
                      1.59680000e+01,9.87700000e+00,4.00000000e+00,3.30000000e+00,2.60000000e+00,
                      2.10000000e+00,1.50000000e+00,1.30000000e+00,1.15000000e+00,1.12300000e+00,
                      1.09700000e+00,1.07100000e+00,1.04500000e+00,1.02000000e+00,9.96000000e-01,
                      9.72000000e-01,9.50000000e-01,9.10000000e-01,8.50000000e-01,7.80000000e-01,
                      6.25000000e-01,5.00000000e-01,4.00000000e-01,3.50000000e-01,3.20000000e-01,
                      3.00000000e-01,2.80000000e-01,2.50000000e-01,2.20000000e-01,1.80000000e-01,
                      1.40000000e-01,1.00000000e-01,8.00000000e-02,6.70000000e-02,5.80000000e-02,
                      5.00000000e-02,4.20000000e-02,3.50000000e-02,3.00000000e-02,2.50000000e-02,
                      2.00000000e-02,1.50000000e-02,1.00000000e-02,5.00000000e-03,1.00000000e-05])
        div = [27, 54]
    E0 = E[:-1]
    E1 = E[1:]
    de = E[:-1] - E[1:]
    E = np.concatenate((E0, E1)).reshape(-1,len(E0)).T.flatten()
    phi = np.concatenate((phi / de, phi / de)).reshape(-1,len(E0)).T.flatten()
    
    plt.loglog(E, phi)
    plt.ylabel('$\phi$')
    plt.xlabel('Energy [eV]')
    
    ymin, ymax = plt.gca().get_ylim()
    
    for d in div:
        x = E0[d] if G in [69, 238] else d
        plt.plot((x, x), (ymin, ymax), 'y--')
    plt.savefig('phi{}.pdf'.format(int(len(phi)/2)))
    #plt.show()
 
def solveEigenProblem(sig_t, sig_s, vsig_f, chi):
    M = np.diag(sig_t) - sig_s
    M = np.linalg.inv(M).dot(np.outer(chi,vsig_f))
    eigs = np.linalg.eig(M)
    i = np.argmax(np.real(eigs[0]))
    k = np.real(eigs[0][i])
    phi = np.real(eigs[1][:,i])
    phi *= -1 if phi[0] < 0 else 1
    return phi, k
    
def getName(G):
    if G == 238:
        return 'uo2XS-{}.h5'
    if G == 69:
        return '69-groupData/uo2XS-{}-69.h5'
    
def runFracUO2(G):
    sig_t, sig_s, vsig_f, chi, div = getData(G)
    d = np.array([0] + div + [G])
    d = max(d[1:] - d[:-1])
    
    phi, k = solveEigenProblem(sig_t, sig_s, vsig_f, chi)
    phi /= norm(phi)
    #plotPhi(phi)
    
    for f in np.linspace(0,1,21)[rank::size]:
        with h5py.File(getName(G).format(f), 'r') as F:
            pat, _ = solveEigenProblem(F['sig_t'][...], F['sig_s'][...], F['vsig_f'][...], F['chi'][...])
            pat /= norm(pat)
        bType = 'mdlp'
        eps = []
        epsk = []
        x = range(d)
        ps = []
        for trunc in x:
            P, K = fixedSourceSolver(sig_t, sig_s, vsig_f, chi, k=k, lamb=0.1, truncate=trunc, bType=bType, bPattern=pat, division=div).solve(silent=True)
            P /= norm(P)
            ps.append(P[:])
            ep = norm(phi-P)
            epk = abs(k - K) / k
            print 'rank = {}, DOF = {}, error = {}'.format(rank, trunc, ep)
            eps.append(ep)
            epsk.append(epk)
        np.savetxt('eigenUO2-{}-{}.dat'.format(f, G), eps)
        np.savetxt('eigenUO2-k-{}-{}.dat'.format(f, G), epsk)
        np.savetxt('eigenUO2-phi-{}-{}.dat'.format(f, G), ps)
        
    if rank == size - 1:
        bType = 'dlp'
        eps = []
        epsk = []
        ps = []
        x = range(d)
        for trunc in x:
            P, K = fixedSourceSolver(sig_t, sig_s, vsig_f, chi, k=k, lamb=0.1, truncate=trunc, bType=bType, bPattern=None, division=div).solve(silent=True)
            P /= norm(P)
            ps.append(P[:])
            ep = norm(phi-P)
            epk = abs(k - K) / k
            print 'rank = {}, DOF = {}, error = {}'.format(rank, trunc, ep)
            eps.append(ep)
            epsk.append(epk)
        np.savetxt('eigenUO2-dlp-{}.dat'.format(G), eps)
        np.savetxt('eigenUO2-k-dlp-{}.dat'.format(G), epsk)
        np.savetxt('eigenUO2-phi-dlp-{}.dat'.format(G), ps)
    
def plotFracStudy(G):
    plt.clf()
    sig_t, sig_s, vsig_f, chi, div = getData(G)
    d = np.array([0] + div + [G])
    d = max(d[1:] - d[:-1])
    x = range(d)
    
    plt.semilogy(x, np.loadtxt('eigenUO2-dlp-{}.dat'.format(G)), label='DLP')
    for f in np.linspace(0,1,11):
        plt.semilogy(x, np.loadtxt('eigenUO2-{}-{}.dat'.format(f, G)), label='mDLP {}'.format(f))
    
    plt.legend(bbox_to_anchor=(0.95, 0.5),ncol=3)
    plt.ylabel('$||\phi-\phi^*||$')
    plt.xlabel('Basis functions included in each course group')
    plt.savefig('eigenUO2_comparison-{}.pdf'.format(G))
    
    plt.semilogy(x, np.loadtxt('eigenUO2-k-dlp-{}.dat'.format(G)), label='DLP')
    for f in np.linspace(0,1,11):
        plt.semilogy(x, np.loadtxt('eigenUO2-k-{}-{}.dat'.format(f, G)), label='mDLP {}'.format(f))
    
    plt.legend(bbox_to_anchor=(0.95, 0.5),ncol=3)
    plt.ylabel('$||k-k^*||$')
    plt.xlabel('Basis functions included in each course group')
    plt.savefig('eigenUO2_comparison-k-{}.pdf'.format(G))
    
def plotBasis(G, bType, phi=None):  
    plt.clf()
    types = ['b-', 'g-', 'k-', 'r-']
    N = 3
    sig_t, sig_s, vsig_f, chi, div = getData(G)
    B = fixedSourceSolver(sig_t, sig_s, vsig_f, chi, truncate=N, bType=bType, bPattern=phi, division=div).basis
    div = div if div is not None else []
    div = [0] + div + [G]
    
    ymin = min([np.min(b) for b in B]) - .1
    ymax = max([np.max(b) for b in B]) + .2
    
    if G == 238:
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
        for i, b in enumerate(B):
            for j, bb in enumerate(b):
                r, bb = barchart(E[div[i]:div[i+1]+1], bb)
                plt.semilogx(r, bb, types[j])
        plt.xlabel('Energy [eV]')
    elif G == 69:
        E = np.array([1.00000000e+07,6.06550000e+06,3.67900000e+06,2.23100000e+06,1.35300000e+06,
                      8.21000000e+05,5.00000000e+05,3.02500000e+05,1.83000000e+05,1.11000000e+05,
                      6.73400000e+04,4.08500000e+04,2.47800000e+04,1.50300000e+04,9.11800000e+03,
                      5.53000000e+03,3.51910000e+03,2.23945000e+03,1.42510000e+03,9.06899000e+02,
                      3.67263000e+02,1.48729000e+02,7.55014000e+01,4.80520000e+01,2.77000000e+01,
                      1.59680000e+01,9.87700000e+00,4.00000000e+00,3.30000000e+00,2.60000000e+00,
                      2.10000000e+00,1.50000000e+00,1.30000000e+00,1.15000000e+00,1.12300000e+00,
                      1.09700000e+00,1.07100000e+00,1.04500000e+00,1.02000000e+00,9.96000000e-01,
                      9.72000000e-01,9.50000000e-01,9.10000000e-01,8.50000000e-01,7.80000000e-01,
                      6.25000000e-01,5.00000000e-01,4.00000000e-01,3.50000000e-01,3.20000000e-01,
                      3.00000000e-01,2.80000000e-01,2.50000000e-01,2.20000000e-01,1.80000000e-01,
                      1.40000000e-01,1.00000000e-01,8.00000000e-02,6.70000000e-02,5.80000000e-02,
                      5.00000000e-02,4.20000000e-02,3.50000000e-02,3.00000000e-02,2.50000000e-02,
                      2.00000000e-02,1.50000000e-02,1.00000000e-02,5.00000000e-03,1.00000000e-05])
        for i, b in enumerate(B):
            for j, bb in enumerate(b):
                r, bb = barchart(E[div[i]:div[i+1]+1], bb)
                plt.semilogx(r, bb, types[j])
        plt.xlabel('Energy [eV]')
    else:
        for i, b in enumerate(B):
            
            for j, bb in enumerate(b):
                r = range(div[i], div[i+1]+1)
                r, bb = barchart(r, bb)
                plt.plot(r, bb, types[j])
        plt.xlabel('group')
    for d in div[1:-1]:
        x = E[d] if G in [238, 69] else d
        plt.plot((x, x), (ymin, ymax), 'y--')
    plt.ylabel('Normalized basis') 
    plt.legend(['basis{}'.format(i) for i in range(N+1)], loc=2, ncol=N+1, mode="expand", borderaxespad=0.5)
    plt.ylim([ymin, ymax])
    plt.savefig('basis{}-{}.pdf'.format(G, bType))
    #plt.show()
    
def barchart(x, y) :
    X = np.zeros(2*len(y))
    Y = np.zeros(2*len(y))
    for i in range(0, len(y)) :
        X[2*i]   = x[i]
        X[2*i+1] = x[i+1]
        Y[2*i]   = y[i]
        Y[2*i+1] = y[i]
    return X, Y

def runThings(G):
    bType = 'dlp'
    
    sig_t, sig_s, vsig_f, chi, div = getData(G)
    
    print 'Eigenvalue Problem'
    M = np.diag(sig_t) - sig_s
    M = np.linalg.inv(M).dot(np.outer(chi,vsig_f))
    eigs = np.linalg.eig(M)
    i = np.argmax(np.real(eigs[0]))
    k = np.real(eigs[0][i])
    phi = np.real(eigs[1][:,i])
    phi *= -1 if phi[0] < 0 else 1
    plotPhi(phi)
    plotBasis(G, bType, phi)
    #plotPhi(phi)
    P, kk = fixedSourceSolver(sig_t, sig_s, vsig_f, chi, lamb=0.01, phi=phi, k=k, truncate=None, bType=bType, bPattern=phi, division=div).solve(silent=False)
    print norm(phi / norm(phi) - P / norm(P))
    print k - kk
        
if __name__ == "__main__":
    G = 238
    runFracUO2(G)
    #plotFracStudy(G)
    #runThings(G)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
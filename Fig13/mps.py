from renormalizer.sbm import SpinBosonDynamics
from renormalizer.model import Model, Op
from renormalizer.mps import Mps, Mpo, MpDm, ThermalProp
from renormalizer.utils.constant import *
from renormalizer.model import basis as ba
from renormalizer.utils import OptimizeConfig, EvolveConfig, CompressConfig, CompressCriteria, EvolveMethod
from renormalizer.utils import log, Quantity

import logging
import itertools
import numpy as np
import scipy

class SpectralDensityFunction:
    """
    the sub-ohmic spectral density function
    """

    def __init__(self, ita, omega_c, beta, omega_limit):
        self.ita = ita
        self.omega_c = omega_c
        self.beta = beta
        self.omega_limit = omega_limit


    def reno(self, omega_l):
        def integrate_func(x):
            return self.func(x) / x**2

        res = scipy.integrate.quad(integrate_func, a=omega_l,
                b=omega_l*1000)
        logger.info(f"integrate: {res[0]}, {res[1]}")
        re = np.exp(-res[0]*2/np.pi)

        return re


    def func(self, omega_value):
        """
        the function of the ohmic spectral density function
        """
        theta = np.arctan(omega_value/self.omega_c)
        return self.ita * np.sin(self.beta * theta) / (1 + omega_value**2/omega_c**2) ** (self.beta / 2)


    def _dos_Wang1(self, A, omega_value):
        """
        Wang's 1st scheme DOS \rho(\omega)
        """

        return A * self.func(omega_value) / omega_value

    def Wang1(self, nb):
        """
        Wang's 1st scheme discretization
        """
        def integrate_func(x):
            return self.func(x) / x
        A = (nb + 1 ) / scipy.integrate.quad(integrate_func, a=0, b=self.omega_limit)[0]
        logger.info(scipy.integrate.quad(integrate_func, a=0, b=self.omega_limit)[0] * 4 / np.pi)
        logger.info(2*self.ita)
        nsamples = int(1e7)
        delta = self.omega_limit / nsamples
        omega_value_big = np.linspace(delta, self.omega_limit, nsamples)
        dos = self._dos_Wang1(A, omega_value_big)
        rho_cumint = np.cumsum(dos) * delta
        diff = (rho_cumint % 1)[1:] - (rho_cumint % 1)[:-1]
        idx = np.where(diff < 0)[0]
        omega_value = omega_value_big[idx]
        logger.info(len(omega_value))
        assert len(omega_value) == nb

        # general form
        c_j2 = 2./np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(A, omega_value)


        return omega_value, c_j2


from renormalizer.utils.log import package_logger
logger = package_logger
dump_dir = "./"
job_name = "test"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")

ita = 5
eps = 0
Delta = 1
omega_c = 0.1
nmodes = 100
beta = 0.5
sdf = SpectralDensityFunction(ita, omega_c, beta, 30*omega_c)

w, c2 = sdf.Wang1(nmodes)
c = np.sqrt(c2)
logger.info(f"w:{w}")
logger.info(f"c:{c}")

reno = sdf.reno(w[-1])
logger.info(f"renormalization constant: {reno}")
Delta *= reno

ham_terms = []

# h_s
ham_terms.extend([Op("sigma_z","spin",factor=eps, qn=0),
        Op("sigma_x","spin",factor=Delta, qn=0)])


# boson energy
for imode in range(nmodes):
    op1 = Op(r"p^2",f"v_{imode}",factor=0.5, qn=0)
    op2 = Op(r"x^2",f"v_{imode}",factor=0.5*w[imode]**2, qn=0)
    ham_terms.extend([op1,op2])

# system-boson coupling
for imode in range(nmodes):
    op = Op(r"sigma_z x", ["spin", f"v_{imode}"],
            factor=c[imode], qn=[0,0])
    ham_terms.append(op)

basis = [ba.BasisHalfSpin("spin",[0,0])]
for imode in range(nmodes):
    basis.append(ba.BasisSHO(f"v_{imode}", w[imode], 10))

model = Model(basis, ham_terms)
evolve_config = EvolveConfig(EvolveMethod.tdvp_ps,
        )
compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=20)
sbm = SpinBosonDynamics(model, evolve_config=evolve_config,
        compress_config=compress_config, dump_dir=dump_dir,
        job_name=job_name)


mps = sbm.latest_mps
model = Model(basis, mps.model.ham_terms)
mpo = Mpo(model)
exp_z = Mpo(model, Op("sigma_z", "spin"))
exp_x = Mpo(model, Op("sigma_x", "spin"))
nsteps = 100
dt = 0.05
# expectations = []
for i in range(nsteps):
    mps = mps.evolve(mpo, dt)
    z = mps.expectation(exp_z)
    x = mps.expectation(exp_x)
    print(z, x)

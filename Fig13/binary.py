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
import sys
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

ita_str = sys.argv[1]  # 010, 025, 100
omega_c_str = sys.argv[2]  # 001, 100
beta_str = sys.argv[3]  # 025, 100
# ita_str = "050"
# omega_c_str = "001"
# beta_str = "050"

ita = int(ita_str) / 10 # 1, 2.5, 5, 10
eps = 0
Delta = 1
omega_c = int(omega_c_str) / 10  # 0.1, 1, 10

beta = int(beta_str) / 100  # 0.25, 0.5, 0.75, 1

from renormalizer.utils.log import package_logger
logger = package_logger
dump_dir = "./Fig3a/"
job_name = f"ps1_binary_ita{ita_str}_omega{omega_c_str}_beta{beta_str}"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")


nmodes = 1000
Ms = 20
sdf = SpectralDensityFunction(ita, omega_c, beta, 300*omega_c)

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

nbas = np.max([16 * c2/w**3, np.ones(nmodes)*4], axis=0)
nbas = np.round(nbas).astype(int)
logger.info(nbas)
basis = [ba.BasisHalfSpin("spin",[0,0])]
for imode in range(nmodes):
    basis.append(ba.BasisSHO(f"v_{imode}", w[imode], int(nbas[imode])))

from collections import deque
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis
from renormalizer.model.basis import BasisDummy
from renormalizer.tn.treebase import approximate_partition

tree_order = 2
basis_vib = basis[1:]
elementary_nodes = []


root = BasisTree.binary_mctdh(basis_vib, contract_primitive=True, contract_label=nbas>Ms, dummy_label="n").root

root.add_child(TreeNodeBasis(basis[:1]))

basis_tree = BasisTree(root)
basis_tree.print(print_function=logger.info)

# basis_tree = BasisTree.linear(basis)
ttno = TTNO(basis_tree, ham_terms)
exp_z = TTNO(basis_tree, Op("sigma_z", "spin"))
exp_x = TTNO(basis_tree, Op("sigma_x", "spin"))
ttns = TTNS(basis_tree)
ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=Ms)
from expand import expand_bond_dimension
ttns = expand_bond_dimension(ttns, ttno, include_ex=False)
logger.info(ttns.bond_dims)
logger.info(ttno.bond_dims)
logger.info(len(ttns))
ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
nsteps = 400
dt = 0.1
expectations = []
for i in range(nsteps):
    ttns = ttns.evolve(ttno, dt)
    z = ttns.expectation(exp_z)
    x = ttns.expectation(exp_x)
    expectations.append((z, x))
    logger.info((z, x))
logger.info(expectations)
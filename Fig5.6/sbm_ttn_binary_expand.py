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

    def __init__(self, alpha, omega_c, s):
        self.alpha = alpha
        self.omega_c = omega_c
        self.s = s

    def reno(self, omega_l):
        def integrate_func(x):
            return self.func(x) / x**2

        res = scipy.integrate.quad(integrate_func, a=omega_l,
                b=self.omega_c*30)
        logger.info(f"integrate: {res[0]}, {res[1]}")
        re = np.exp(-res[0]*2/np.pi)

        return re


    def adiabatic_renormalization(self, delta, p: float):
        """
        the cut-off omega_l is p*delta
        """
        loop = 0
        re = 1.
        while loop < 50:
            re_old = re
            omega_l = delta * re * p
            def integrate_func(x):
                return self.func(x) / x**2

            res = scipy.integrate.quad(integrate_func, a=omega_l,
                    b=self.omega_c*30)
            logger.info(f"integrate: {res[0]}, {res[1]}")
            re = np.exp(-res[0]*2/np.pi)
            loop += 1
            logger.info(f"re, {re_old}, {re}")
            if np.allclose(re, re_old):
                break

        return delta*re, delta*re*p


    def func(self, omega_value):
        """
        the function of the ohmic spectral density function
        """
        return np.pi / 2. * self.alpha * omega_value**self.s *\
                self.omega_c**(1-self.s) * np.exp(-omega_value / self.omega_c)

    @staticmethod
    def post_process(omega_value, c_j2, ifsort):
        displacement_array = np.sqrt(c_j2) / omega_value ** 2
        if ifsort:
            idx = np.argsort(c_j2 / omega_value)[::-1]
        else:
            idx = np.arange(len(omega_value))
        omega_list = []
        displacement_list = []
        for i in idx:
            omega_list.append(Quantity(omega_value[i]))
            displacement_list.append(Quantity(displacement_array[i]))
        return omega_list, displacement_list

    def _dos_Wang1(self, nb, omega_value):
        """
        Wang's 1st scheme DOS \rho(\omega)
        """
        return (nb + 1) / self.omega_c * np.exp(-omega_value / self.omega_c)

    def Wang1(self, nb):
        """
        Wang's 1st scheme discretization
        """
        omega_value = np.array([-np.log(-float(j) / (nb + 1) + 1.) * self.omega_c for j in range(1, nb + 1, 1)])

        # general form
        c_j2 = 2./np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(nb, omega_value)


        return omega_value, c_j2


    def trapz(self, nb, x0, x1):
        dw = (x1 - x0) / float(nb)
        xlist = [x0 + i * dw for i in range(nb + 1)]
        omega_value = np.array([(xlist[i] + xlist[i + 1]) / 2. for i in range(nb)])
        c_j2 = np.array([(self.func(xlist[i]) + self.func(xlist[i + 1])) / 2 for i in range(nb)]) * 2. / np.pi * omega_value * dw

        return omega_value, c_j2


from renormalizer.utils.log import package_logger
#logger = logging.getLogger(__name__)
logger = package_logger
dump_dir = "./"
job_name = "test"  ####################
#log.register_file_output(dump_dir+job_name+".log", mode="w")

import sys

alpha = float(sys.argv[1]) / 100
logger.info(f"alpha:{alpha}")
eps = 0
Delta = 1
omega_c = 20
nmodes = 1000
s = 0.5
sdf = SpectralDensityFunction(alpha, omega_c, s)

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

from collections import deque
from renormalizer.tn import BasisTree, TTNO, TTNS, TreeNodeBasis
from renormalizer.model.basis import BasisDummy

tree_order = 2
basis_vib = basis[1:]
root = BasisTree.binary_mctdh(basis_vib).root

root.add_child(TreeNodeBasis(basis[:1]))

basis_tree = BasisTree(root)
basis_tree.print()

# basis_tree = BasisTree.linear(basis)
ttno = TTNO(basis_tree, ham_terms)
exp_z = TTNO(basis_tree, Op("sigma_z", "spin"))
exp_x = TTNO(basis_tree, Op("sigma_x", "spin"))
ttns = TTNS(basis_tree)
ttns.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=20)
from expand import expand_bond_dimension
ttns = expand_bond_dimension(ttns, ttno, include_ex=False)
logger.info(ttns.bond_dims)
logger.info(ttno.bond_dims)
logger.info(len(ttns))
ttns.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
nsteps = 800
dt = 0.05
expectations = []
for i in range(nsteps):
    ttns = ttns.evolve(ttno, dt)
    z = ttns.expectation(exp_z)
    x = ttns.expectation(exp_x)
    expectations.append((z, x))
    print(z, x)
print(expectations)

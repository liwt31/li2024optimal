import sys

from renormalizer import BasisHalfSpin, Op, Quantity, Mpo, Model, Mps, BasisSHO, BasisDummy
from renormalizer.utils.configs import EvolveMethod, EvolveConfig, CompressConfig, CompressCriteria
from renormalizer.utils import constant
from renormalizer.tn import BasisTree, TreeNodeBasis, TTNS, TTNO

import numpy as np
import scipy

from renormalizer.utils.log import package_logger as logger
from renormalizer.utils import log

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


Ms_str = sys.argv[1]  # 8 16 24 32
initial_str = sys.argv[2] # 1 0
temperature_str = sys.argv[3] # 000 100 200 300

dump_dir = "./more_modes2/"
job_name = f"Ms{Ms_str}_i{initial_str}_temperature{temperature_str}"  ####################
log.register_file_output(dump_dir+job_name+".log", mode="w")

Ms = int(Ms_str)

n_ph_mode = 1000
omega_c = Quantity(500, "cm-1").as_au()
ita = Quantity(2000, "cm-1").as_au() / 2
beta = 0.5
upper_limit = Quantity(1, "eV").as_au() * 10
logger.info(("phonon parameters", omega_c, ita, beta, upper_limit))
sdf = SpectralDensityFunction(ita, omega_c, beta, upper_limit)
w, c2 = sdf.Wang1(n_ph_mode)
c = np.sqrt(c2)
logger.info(w)
logger.info(c)

reno = sdf.reno(w[-1])
logger.info(f"renormalization constant: {reno}")

temperature = Quantity(int(temperature_str), "K").to_beta()
logger.info(f"temperature beta: {temperature}")

n_e_mode = 320

beta_e = Quantity(1, "eV").as_au() * reno
alpha_e = Quantity(0.2, "eV").as_au() * reno
v = .1 * reno
mu_l = Quantity(v/2, "eV").as_au()
mu_r = Quantity(-v/2, "eV").as_au()


e_k = np.arange(1, n_e_mode + 1) / (n_e_mode + 1) * 4 * beta_e - 2 * beta_e
rho_e = 1 / (e_k[1] - e_k[0])
e_k_l = e_k - mu_l
e_k_r = e_k - mu_r

mode_with_e = [(f"L{i}", e) for i, e in enumerate(e_k_l)] + [(f"R{i}", e) for i, e in enumerate(e_k_r)]
mode_with_e.sort(key=lambda x:x[1])
logger.info(mode_with_e)

basis = []
first_positive = True
for mode, e in mode_with_e:
    if e > 0 and first_positive:
        first_positive = False
        basis.append(BasisHalfSpin("s"))
    basis.append(BasisHalfSpin((mode, "p")))
    basis.append(BasisHalfSpin((mode, "q")))

dofs = [b.dofs[0] for b in basis]
logger.info(dofs)
s_idx = dofs.index("s")
basis_tree_l = BasisTree.binary_mctdh(basis[:s_idx], dummy_label="EL-dummy")
basis_tree_r = BasisTree.binary_mctdh(basis[s_idx+1:], dummy_label="ER-dummy")


ham_terms = []
i_l_terms = []
i_r_terms = []
for mode, e in mode_with_e:
    if mode[0] == "L":
        mu = mu_l
        i_terms = i_l_terms
    else:
        assert mode[0] == "R"
        mu = mu_r
        i_terms = i_r_terms

    ham_terms.append(Op("+ -", (mode, "p"), e + mu))
    ham_terms.append(Op("+ -", (mode, "q"), -(e + mu)))
    v2 = alpha_e ** 2 / beta_e ** 2 * np.sqrt(4*beta_e**2 - (e + mu)**2) / 2 / np.pi / rho_e
    v = np.sqrt(v2)
    theta = np.arctan(np.exp(-temperature*e/2))
    idx = dofs.index((mode, "p"))
    if idx < s_idx:
        z_idx = list(range(idx+1, s_idx))
    else:
        assert s_idx < idx
        z_idx = list(range(s_idx+1, idx))
    z_dofs = [dofs[i] for i in z_idx]
    op1 = Op("+ " + "Z " * len(z_idx) + "-", [(mode, "p")] + z_dofs + ["s"], v * np.cos(theta))
    op2 = Op("- " + "Z " * len(z_idx) + "+", [(mode, "p")] + z_dofs + ["s"], v * np.cos(theta))
    idx = dofs.index((mode, "q"))
    if idx < s_idx:
        z_idx = list(range(idx+1, s_idx))
    else:
        assert s_idx < idx
        z_idx = list(range(s_idx+1, idx))
    z_dofs = [dofs[i] for i in z_idx]
    op3 = Op("- " + "Z " * len(z_idx) + "-", [(mode, "q")] + z_dofs + ["s"], v * np.sin(theta))
    op4 = Op("+ " + "Z " * len(z_idx) + "+", [(mode, "q")] + z_dofs + ["s"], v * np.sin(theta))
    ham_terms.extend([op1, op2, op3, op4])
    # move 1j to expectation
    i_terms.extend(op2 - op1 + op4 - op3)

initial_occupied = initial_str == "1"

if initial_occupied:
    ham_terms.append(Op("+ -", "s", qn=[0, 0], factor=-4*(c**2/w**2).sum()))

# vibrations at last

# boson energy
for imode in range(n_ph_mode):
    op1 = Op(r"p^2",f"v_{imode}_p",factor=0.5, qn=0)
    op2 = Op(r"x^2",f"v_{imode}_p",factor=0.5*w[imode]**2, qn=0)
    ham_terms.extend([op1,op2])
    op1 = Op(r"p^2",f"v_{imode}_q",factor=-0.5, qn=0)
    op2 = Op(r"x^2",f"v_{imode}_q",factor=-0.5*w[imode]**2, qn=0)
    ham_terms.extend([op1,op2])

theta_array = np.arctanh(np.exp(-w * temperature / 2))
# system-boson coupling
for imode in range(n_ph_mode):
    sys_op = Op("+ -", "s", qn=[0, 0])
    if initial_occupied:
        sys_op = sys_op - Op.identity("s")

    theta = theta_array[imode]
    op1 = sys_op * Op(r"x", f"v_{imode}_p", factor=2 * c[imode] * np.cosh(theta), qn=[0])
    op2 = sys_op * Op(r"x", f"v_{imode}_q", factor=2 * c[imode] * np.sinh(theta), qn=[0])
    ham_terms.extend(op1 + op2)

nbas = np.max([16 * c2/w**3 * np.cosh(theta_array)**2, np.ones(n_ph_mode)*4], axis=0)
nbas = np.round(nbas).astype(int)
nbas = np.min([nbas, np.ones(n_ph_mode)*512], axis=0)
logger.info(nbas)
basis_list_phonon = []
for imode in range(n_ph_mode):
    basis_list_phonon.append(BasisSHO(f"v_{imode}_p", w[imode], int(nbas[imode])))
    basis_list_phonon.append(BasisSHO(f"v_{imode}_q", w[imode], int(nbas[imode])))

labels = np.array([[nbas>Ms], [nbas>Ms]]).T.ravel()
basis_tree_phonon = BasisTree.binary_mctdh(basis_list_phonon, contract_primitive=True, contract_label=labels, dummy_label="phonon-dummy")
node1 = TreeNodeBasis([BasisDummy("dummy")])
node1.add_child([basis_tree_l.root, basis_tree_r.root])
node2 = TreeNodeBasis([basis[s_idx]])
node2.add_child([node1, basis_tree_phonon.root])
basis_tree = BasisTree(node2)
basis_tree.print(logger.info)


#model = Model(basis, ham_terms)
ttno = TTNO(basis_tree, ham_terms)
i_l_mpo = TTNO(basis_tree, i_l_terms)
i_r_mpo = TTNO(basis_tree, i_r_terms)
#n_l_mpo = TTNO(basis_tree, terms=[Op("+ -", f"L{i}") for i in range(n_e_mode)])
#n_r_mpo = TTNO(basis_tree, terms=[Op("+ -", f"R{i}") for i in range(n_e_mode)])
n_s_mpo = TTNO(basis_tree, terms=Op("+ -", "s"))
ttno.print_shape(False, logger.info)
i_l_mpo.print_shape(False, logger.info)
i_r_mpo.print_shape(False, logger.info)
# n_r_mpo.print_shape(False, logger.info)
# n_s_mpo.print_shape(False, logger.info)
# 0 - [1, 0] (spin up) means occupation, 1 - [0, 1] (spin down) means unoccupation
# initial condition is taken care of in the Hamiltonian
condition = {dofs[i]:1 for i in range(len(dofs))}
if initial_occupied:
    condition["s"] = 0
else:
    condition["s"] = 1

mps = TTNS(basis_tree, condition=condition)
mps.compress_config = CompressConfig(CompressCriteria.fixed, max_bonddim=Ms)
from expand import expand_bond_dimension
mps = expand_bond_dimension(mps, ttno, include_ex=False)
mps.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
mps.print_shape(print_function=logger.info, full=False)

step = 0.5 * constant.fs2au
# step = 5
nsteps = 200
au2muA = 6.623618237510e3
i = 0
current_list = []
while True:
    i_l = (1j * mps.expectation(i_l_mpo)).real
    i_r = (1j * mps.expectation(i_r_mpo)).real
    # n_l = mps.expectation(n_l_mpo)
    # n_r = mps.expectation(n_r_mpo)
    n_s = mps.expectation(n_s_mpo)
    logger.info((i, mps.bond_dims))
    current = (i_r-i_l)/2*au2muA
    #logger.info((n_l, n_r, n_s, i_l*au2muA, i_r*au2muA, current))
    logger.info((n_s, i_l*au2muA, i_r*au2muA, current))
    current_list.append(current)
    i += 1
    if i == nsteps:
        break
    if i > 0:
        mps.evolve_config = EvolveConfig(EvolveMethod.tdvp_ps)
    mps = mps.evolve(ttno, step)
logger.info(current_list)

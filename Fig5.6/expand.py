from typing import List
from renormalizer.utils.log import package_logger as logger
from renormalizer.utils import CompressCriteria
from renormalizer.mps.lib import compressed_sum


def expand_bond_dimension(self, hint_mpo=None, coef=1e-10, include_ex=True):
    """
    expand bond dimension as required in compress_config
    """
    # expander m target
    m_target = self.compress_config.bond_dim_max_value - self.bond_dims_mean
    # will be restored at exit
    self.compress_config.bond_dim_max_value = m_target
    if self.compress_config.criteria is not CompressCriteria.fixed:
        logger.warning("Setting compress criteria to fixed")
        self.compress_config.criteria = CompressCriteria.fixed
    logger.debug(f"target for expander: {m_target}")
    if hint_mpo is None:
        expander = self.__class__.random(self.model, 1, m_target)
    else:
        # fill states related to `hint_mpo`
        logger.debug(
            f"average bond dimension of hint mpo: {hint_mpo.bond_dims_mean}"
        )
        # in case of localized `self`
        if include_ex:
            if self.is_mps:
                ex_state = self.ground_state(self.model, False)
                # for self.qntot >= 1
                assert self.model.qn_size == 1  # otherwise not supported
                for i in range(self.qntot[0]):
                    ex_state = Mpo.onsite(self.model, r"a^\dagger") @ ex_state
            elif self.is_mpdm:
                assert self.qntot == 1
                ex_state = self.max_entangled_ex(self.model)
            else:
                assert False
            ex_state.compress_config = self.compress_config
            ex_state.move_qnidx(self.qnidx)
            ex_state.to_right = self.to_right
            lastone = self + ex_state

        else:
            lastone = self
        expander_list: List["MatrixProduct"] = []
        cumulated_m = 0
        while True:
            lastone.compress_config.criteria = CompressCriteria.fixed
            expander_list.append(lastone)
            expander = compressed_sum(expander_list)
            if cumulated_m == expander.bond_dims_mean:
                # probably a small system, the required bond dimension can't be reached
                break
            cumulated_m = expander.bond_dims_mean
            logger.debug(
                f"cumulated bond dimension: {cumulated_m}. lastone bond dimension: {lastone.bond_dims}"
            )
            if m_target < cumulated_m:
                break
            if m_target < 0.8 * (lastone.bond_dims_mean * hint_mpo.bond_dims_mean):
                lastone = lastone.canonicalise().compress(
                    m_target // hint_mpo.bond_dims_mean + 1
                )
            lastone = (hint_mpo @ lastone).normalize("mps_and_coeff")
    logger.debug(f"expander bond dimension: {expander.bond_dims}")
    self.compress_config.bond_dim_max_value += self.bond_dims_mean
    return (self + expander.scale(coef * self.norm, inplace=True)).canonicalise().canonicalise().normalize(
        "mps_norm_to_coeff")
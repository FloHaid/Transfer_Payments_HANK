"""Define HA-one and HA-two model blocks."""

import numpy as np
import sequence_jacobian as sj
#from .two_account_calvo import twoasset_calvo, calvo, make_grids_twoassets
from base import calibration

"""HA-one model"""

make_grids = sj.hetblocks.hh_sim.make_grids

def income(Z, e_grid):
    y = Z * e_grid
    return y

#ha_one = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids, income])






"""HA-one with transfers from hank file """
def make_grids_pdf(rho_e, sd_e, n_e, min_a, max_a, n_a):
    e_grid, e_pdf, Pi = sj.grids.markov_rouwenhorst(rho_e, sd_e, n_e)
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    return e_grid, e_pdf, Pi, a_grid
    
def transfers(e_pdf, T, e_grid):
    # hardwired incidence rules are proportional to skill; scale does not matter 
    tax = T / np.sum(e_pdf * e_grid) * e_grid
    return tax

def income_taxed(Z, e_grid, tax, pi):
    y = Z * e_grid - tax * (1+pi)
    return y

ha_one = sj.hetblocks.hh_sim.hh.add_hetinputs([make_grids_pdf, transfers, income_taxed])







"""Quantitative case, with earnings-weighted average marginal utility, UCE, as extra hetoutput"""

def compute_uce_one(c,e_grid,eis):
    uce = c ** (-1 / eis) * e_grid[:, np.newaxis]
    return uce




ha_one_uce = sj.create_model([ha_one], name='One Asset Model')
ha_one_uce = ha_one.add_hetoutputs([compute_uce_one])

def get_all_quant(params):
    # two differences from before: we have the "UCE" output, and we use "rpost" rather than "r" as input
    calib_ha_one, calib_ha_two = calibration.get_ha_calibrations()
    hh_het, ss_het = {}, {}
    hh_het['HA-hi-liq'], ss_het['HA-hi-liq'] = ha_one_uce.remap({'r': 'rpost'}), ha_one_uce.steady_state({**calib_ha_one, **params['HA-hi-liq']})
    hh_het['HA-one'], ss_het['HA-one'] = ha_one_uce.remap({'r': 'rpost'}), ha_one_uce.steady_state({**calib_ha_one, **params['HA-one']})

    return hh_het, ss_het


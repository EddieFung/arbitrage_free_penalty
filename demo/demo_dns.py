import numpy as np

from demo import data_simulate
from model import nelson_siegel as ns
from model import afns

np.random.seed(0)
data = data_simulate.DataSimulator()

ns_model = ns.DynamicNelsonSiegel(
    decay_rate=data.decay_rate,
    maturities=data.maturities,
    delta_t=data.delta_t
)
ns_model.inference(data.df, 25000)
ns_filter = ns_model.specify_filter()

afns_model = afns.AFNS(
    maturities=data.maturities,
    delta_t=data.delta_t
)
afns_model.inference(data.df, 25000)
afns_filter = afns_model.specify_filter()


print("Comparison of parameter estimates on DNS:")
print("Theta: {0} vs {1}".format(data.theta, ns_model._theta))
print("Transition matrix diagonal: {0} vs {1}".format(
    np.diag(data.transition_matrix), 
    np.diag(ns_filter.F)
))
print("log-observation sd: {0} vs {1}".format(
    np.log(np.diag(data.observation_std)[:5]), 
    ns_model._log_obs_sd[:5]
))
print("")
print("-----------------------------------------------")
print("Comparison of parameter estimates on AFNS:")
print("Decay rate: {0} vs {1}".format(
    data.decay_rate, 
    np.exp(afns_model._log_rate)
))
print("Theta: {0} vs {1}".format(data.theta, afns_model._theta))
print("Transition matrix diagonal: {0} vs {1}".format(
    np.diag(data.transition_matrix), 
    np.diag(afns_filter.F)
))
print("log-observation sd: {0} vs {1}".format(
    np.log(np.diag(data.observation_std)[:5]), 
    afns_model._log_obs_sd[:5]
))
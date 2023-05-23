from pyddm import Fittable
import pyddm as ddm
import numpy as np
from scipy import stats
from pyddm import InitialCondition
import pyddm.plot
class DriftTTADistance(ddm.models.Drift):
    name = "Drift dynamically depends on the real-time values of distance and TTA"
    required_parameters = ["alpha", "beta_d", "theta"]  # <-- Parameters we want to include in the model
    required_conditions = ["Condition_Dist", "v_start_cluster"]  # <-- Task parameters ("conditions"). Should be the same
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        #Obtained from 01_kinematic_parameters.py
        a_ego = 0.976
        a_ego_tta = 1.0
        a_opp = 8.723
        a_opp_tta = 10
        t_crit = 0
        t_crit_tta = 1.228
        v_opp = 0.278
        a_opp_tta = np.where(t < t_crit_tta, 0, a_opp_tta)

        d = conditions["Condition_Dist"] - conditions["v_start_cluster"] * t - 0.5 * a_ego * t ** 2 - 0.5 * a_opp * (
                t - t_crit) ** 2 - v_opp * (t - t_crit)
        tta = d / (conditions["v_start_cluster"] + a_ego_tta * t + a_opp_tta * (t-t_crit_tta) + 0.0001)
        # Calculate and return drift
        return self.alpha * (self.beta_tta * tta + self.beta_d * d - self.theta)

class DriftTTADistanceVel(ddm.models.Drift):
    name = "Drift dynamically depends on the real-time values of distance and TTA, and initial velocity v0"
    required_parameters = ["alpha", "beta_d", "beta_v", "theta"]
    required_conditions = ["Condition_Dist", "v_start_cluster"]  # <-- Task parameters ("conditions"). Should be the same
    beta_tta = 1.0

    def get_drift(self, t, conditions, **kwargs):
        # Obtained from 01_kinematic_parameters.py
        a_ego = 0.976
        a_ego_tta = 1.0
        a_opp = 8.723
        a_opp_tta = 10
        t_crit = 0
        t_crit_tta = 1.228
        v_opp = 0.278
        a_opp_tta = np.where(t < t_crit_tta, 0, a_opp_tta)

        d = conditions["Condition_Dist"] - conditions["v_start_cluster"] * t - 0.5 * a_ego * t ** 2 - 0.5 * a_opp * (
                t - t_crit) ** 2 - v_opp * (t - t_crit)
        tta = d / (conditions["v_start_cluster"] + a_ego_tta * t + a_opp_tta * (t-t_crit_tta) + 0.0001)
        v0 = conditions["v_start_cluster"]
        # Calculate and return drift
        return self.alpha * (self.beta_tta * tta + self.beta_d * d + self.beta_v * v0 - self.theta)


class BoundCollapsingTta(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with TTA"
    required_parameters = ["b_0", "k", "tta_crit"]
    required_conditions = ["Condition_Dist", "v_start_cluster"]
    beta_tta = 1
    def get_bound(self, t, conditions, **kwargs):
        # Obtained from 01_kinematic_parameters.py
         a_ego = 0.976
         a_ego_tta = 1.0
         a_opp = 8.723
         a_opp_tta = 10
         t_crit = 0
         t_crit_tta = 1.228
         v_opp = 0.278
         a_opp_tta = np.where(t < t_crit_tta, 0, a_opp_tta)

         d = conditions["Condition_Dist"] - conditions["v_start_cluster"] * t - 0.5 * a_ego * t ** 2 - 0.5 * a_opp * (
            t - t_crit) ** 2 - v_opp * (t - t_crit)
         tta = d / (conditions["v_start_cluster"] + a_ego_tta * t + a_opp_tta * (t - t_crit_tta) + 0.0001)
         return self.b_0 / (1 + np.exp(-self.k * (tta - self.tta_crit)))

class BoundCollapsingTtaDist(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with TTA and distance"
    required_parameters = ["b_0", "k", "beta_d", "theta"]
    required_conditions = ["Condition_Dist", "v_start_cluster"]
    beta_tta = 1
    def get_bound(self, t, conditions, **kwargs):
        # Obtained from 01_kinematic_parameters.py
         a_ego = 0.976
         a_ego_tta = 1.0
         a_opp = 8.723
         a_opp_tta = 10
         t_crit = 0
         t_crit_tta = 1.228
         v_opp = 0.278
         a_opp_tta = np.where(t < t_crit_tta, 0, a_opp_tta)

         d = conditions["Condition_Dist"] - conditions["v_start_cluster"] * t - 0.5 * a_ego * t ** 2 - 0.5 * a_opp * (
            t - t_crit) ** 2 - v_opp * (t - t_crit)
         tta = d / (conditions["v_start_cluster"] + a_ego_tta * t + a_opp_tta * (t - t_crit_tta) + 0.0001)
         return self.b_0 / (1 + np.exp(-self.k * (self.beta_tta * tta + self.beta_d * d - self.theta)))
class BoundCollapsingTtaDistVel(pyddm.models.Bound):
    name = "Bounds dynamically collapsing with TTA, distance and initial velocity"
    #required_parameters = ["b_0", "k", "tta_crit"]
    required_parameters = ["b_0", "k", "beta_d", "beta_v", "theta"]
    required_conditions = ["Condition_Dist", "v_start_cluster"]
    beta_tta = 1
    def get_bound(self, t, conditions, **kwargs):
        # Obtained from 01_kinematic_parameters.py
         a_ego = 0.976
         a_ego_tta = 1.0
         a_opp = 8.723
         a_opp_tta = 10
         t_crit = 0
         t_crit_tta = 1.228
         v_opp = 0.278
         a_opp_tta = np.where(t < t_crit_tta, 0, a_opp_tta)

         d = conditions["Condition_Dist"] - conditions["v_start_cluster"] * t - 0.5 * a_ego * t ** 2 - 0.5 * a_opp * (
            t - t_crit) ** 2 - v_opp * (t - t_crit)
         tta = d / (conditions["v_start_cluster"] + a_ego_tta * t + a_opp_tta * (t - t_crit_tta) + 0.0001)
         v0 = conditions["v_start_cluster"]
         return self.b_0 / (1 + np.exp(-self.k * (self.beta_tta * tta + self.beta_d * d + self.beta_v * v0 - self.theta)))
class OverlayNonDecisionGaussian(ddm.Overlay):
  """ Courtesy of the pyddm cookbook """
  name = "Add a Gaussian-distributed non-decision time"
  required_parameters = ["ndt_location", "ndt_scale"]




  def apply(self, solution):
      # Extract components of the solution object for convenience
      corr = solution.corr
      err = solution.err
      dt = solution.model.dt
      # Create the weights for different timepoints
      times = np.asarray(list(range(-len(corr), len(corr)))) * dt
      weights = stats.norm(scale=self.ndt_scale, loc=self.ndt_location).pdf(times)
      if np.sum(weights) > 0:
          weights /= np.sum(weights)  # Ensure it integrates to 1
      newcorr = np.convolve(weights, corr, mode="full")[len(corr):(2 * len(corr))]
      newerr = np.convolve(weights, err, mode="full")[len(corr):(2 * len(corr))]
      return ddm.Solution(newcorr, newerr, solution.model,
                            solution.conditions, solution.undec)


class ICLogistic(InitialCondition): #with the courtesy to pyddm cookbook
    name = "A side-biased starting point depending on initial velocity"
    required_parameters = ["b_z", "theta_z"]
    required_conditions = ["v_start_cluster"]
    v_max = 16 #df_rt["v_start"].max()
    v_min = 6 #df_rt["v_start"].min()
    def get_IC(self, x, dx, conditions):
        x0 = 2 / (1 + np.exp(-self.b_z*(conditions["v_start_cluster"] - self.theta_z))) - 0.5
        if not -1 <= x0 <= 1:
            #x0 = np.clip(x0, 0, 1)
            raise ValueError("x0 must be between -1 and 1")
        shift_i = int((len(x)-1)*x0)

        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=x0*2*B.
        return pdf

class ICConstant(InitialCondition): #with the courtesy to pyddm cookbook
    name = "A constant side-biased starting point"
    required_parameters = ["Z"]
    required_conditions = ["v_start_cluster"]
    def get_IC(self, x, dx, conditions):
        x0 = self.Z/2 + 0.5
        shift_i = int((len(x)-1)*x0)
        assert shift_i >= 0 and shift_i < len(x), "Invalid initial conditions"
        pdf = np.zeros(len(x))
        pdf[shift_i] = 1. # Initial condition at x=x0*2*B.
        return pdf

class ModelOutOfTheBox:
  T_dur = 4.0
  param_names = ["alpha", "beta_d", "theta", "b_0", "k", "tta_crit", "Z", "ndt_location", "ndt_scale"]
  def __init__(self):

      self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))
      self.drift = DriftTTADistance(alpha=Fittable(minval=0, maxval=5.0),
                                              beta_d=Fittable(minval=0, maxval=1),
                                              theta=Fittable(minval=4, maxval=80))
      self.IC = ICConstant(Z=Fittable(minval=-1, maxval=1))
      self.bound = BoundCollapsingTta(b_0=Fittable(minval=0.5, maxval=5.0),
                                      k=Fittable(minval=0.01, maxval=2.0),
                                      tta_crit=Fittable(minval=0.1, maxval=5.0))
      self.model = pyddm.Model(name="Model 1", drift=self.drift,
                               noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                               overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)

class ModelOutOfTheBox_Bias:
  T_dur = 4.0
  param_names = ["alpha", "beta_d", "theta", "b_0", "k", "tta_crit", "b_z", "theta_z", "ndt_location", "ndt_scale"]
  def __init__(self):

      self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))
      self.drift = DriftTTADistance(alpha=Fittable(minval=0, maxval=5.0),
                                              beta_d=Fittable(minval=0, maxval=1),
                                              theta=Fittable(minval=4, maxval=80))
      self.IC = ICLogistic(b_z=Fittable(minval=0.01, maxval=0.14),
                           theta_z=Fittable(minval=4, maxval=16))
      self.bound = BoundCollapsingTta(b_0=Fittable(minval=0.5, maxval=5.0),
                                      k=Fittable(minval=0.01, maxval=2.0),
                                      tta_crit=Fittable(minval=0.1, maxval=5.0))
      self.model = pyddm.Model(name="Model 2", drift=self.drift,
                               noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                               overlay=self.overlay,IC=self.IC, T_dur=self.T_dur)

class ModelTTADistVel:
  T_dur = 4.0
  param_names = ["alpha", "beta_d", "beta_v", "theta", "b_0", "k", "tta_crit", "Z", "ndt_location", "ndt_scale"]
  def __init__(self):

      self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))
      self.drift = DriftTTADistanceVel(alpha=Fittable(minval=0, maxval=5.0),
                                              beta_d=Fittable(minval=0, maxval=1),
                                              beta_v=Fittable(minval=0, maxval=5),
                                              theta=Fittable(minval=4, maxval=80))
      self.IC = ICConstant(Z=Fittable(minval=-1, maxval=1))
      self.bound = BoundCollapsingTta(b_0=Fittable(minval=0.5, maxval=5.0),
                                      k=Fittable(minval=0.01, maxval=2.0),
                                      tta_crit=Fittable(minval=0.1, maxval=5.0))
      self.model = pyddm.Model(name="Model 3", drift=self.drift,
                               noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                               overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)

class ModelTTADistVel_Bias:
  T_dur = 4.0
  param_names = ["alpha", "beta_d", "beta_v", "theta", "b_0", "k", "tta_crit", "b_z", "theta_z", "ndt_location", "ndt_scale"]
  def __init__(self):

      self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))
      self.drift = DriftTTADistanceVel(alpha=Fittable(minval=0, maxval=5.0),
                                              beta_d=Fittable(minval=0, maxval=1),
                                              beta_v=Fittable(minval=0, maxval=5),
                                              theta=Fittable(minval=4, maxval=80))
      self.IC = ICLogistic(b_z=Fittable(minval=0.01, maxval=0.14),
                           theta_z=Fittable(minval=4, maxval=16))
      self.bound = BoundCollapsingTta(b_0=Fittable(minval=0.5, maxval=5.0),
                                      k=Fittable(minval=0.01, maxval=2.0),
                                      tta_crit=Fittable(minval=0.1, maxval=5.0))
      self.model = pyddm.Model(name="Model 4", drift=self.drift,
                               noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                               overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)

class Model_Drift_in_Bound:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "theta", "b_0", "k", "Z", "ndt_location","ndt_scale"]

    def __init__(self):
        beta_d_2 = Fittable(minval=0, maxval=1)
        theta_2 = Fittable(minval=4, maxval=80)
        self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftTTADistance(alpha=Fittable(minval=0, maxval=5.0),
                                                beta_d=beta_d_2,
                                                theta=theta_2)
        self.IC = ICConstant(Z=Fittable(minval=-1, maxval=1))
        self.bound = BoundCollapsingTtaDist(b_0=Fittable(minval=0.5, maxval=5.0),
                                       k=Fittable(minval=0.01, maxval=2.0),
                                      beta_d=beta_d_2, theta=theta_2)
        self.model = pyddm.Model(name="Model 5", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)

class Model_Drift_in_Bound_Bias:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "theta", "b_0", "k", "b_z", "theta_z", "ndt_location","ndt_scale"]

    def __init__(self):
        beta_d_2 = Fittable(minval=0, maxval=1)
        theta_2 = Fittable(minval=4, maxval=80)
        self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftTTADistance(alpha=Fittable(minval=0, maxval=5.0),
                                                beta_d=beta_d_2,
                                                theta=theta_2)
        self.IC = ICLogistic(b_z=Fittable(minval=0.01, maxval=0.14),
                             theta_z=Fittable(minval=4, maxval=16))
        self.bound = BoundCollapsingTtaDist(b_0=Fittable(minval=0.5, maxval=5.0),
                                       k=Fittable(minval=0.01, maxval=2.0),
                                      beta_d=beta_d_2, theta=theta_2)
        self.model = pyddm.Model(name="Model 6", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay,IC=self.IC, T_dur=self.T_dur)
class Model_Vel_Drift_in_Bound:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "beta_v", "theta", "b_0", "k", "Z", "ndt_location","ndt_scale"]

    def __init__(self):
        beta_d_2 = Fittable(minval=0, maxval=1)
        beta_v_2 = Fittable(minval=0, maxval=5)
        theta_2 = Fittable(minval=4, maxval=80)
        self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftTTADistanceVel(alpha=Fittable(minval=0, maxval=5.0),
                                                beta_d=beta_d_2,
                                                beta_v=beta_v_2,
                                                theta=theta_2)
        self.IC = ICConstant(Z=Fittable(minval=-1, maxval=1))
        self.bound = BoundCollapsingTtaDistVel(b_0=Fittable(minval=0.5, maxval=5.0),
                                       k=Fittable(minval=0.01, maxval=2.0),
                                      beta_d=beta_d_2, beta_v=beta_v_2, theta=theta_2)
        self.model = pyddm.Model(name="Model 7", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)
class Model_Vel_Drift_in_Bound_Bias:
    T_dur = 4.0
    param_names = ["alpha", "beta_d", "beta_v", "theta", "b_0", "k", "b_z", "theta_z", "ndt_location","ndt_scale"]

    def __init__(self):
        beta_d_2 = Fittable(minval=0, maxval=1)
        beta_v_2 = Fittable(minval=0, maxval=5)
        theta_2 = Fittable(minval=4, maxval=80)
        self.overlay = OverlayNonDecisionGaussian(ndt_location=ddm.Fittable(minval=0, maxval=2.0),
                                                  ndt_scale=ddm.Fittable(minval=0.001, maxval=0.5))

        self.drift = DriftTTADistanceVel(alpha=Fittable(minval=0, maxval=5.0),
                                                beta_d=beta_d_2,
                                                beta_v=beta_v_2,
                                                theta=theta_2)
        self.IC = ICLogistic(b_z=Fittable(minval=0.01, maxval=0.14),
                             theta_z=Fittable(minval=4, maxval=16))
        self.bound = BoundCollapsingTtaDistVel(b_0=Fittable(minval=0.5, maxval=5.0),
                                       k=Fittable(minval=0.01, maxval=2.0),
                                      beta_d=beta_d_2, beta_v=beta_v_2, theta=theta_2)
        self.model = pyddm.Model(name="Model 8", drift=self.drift,
                                 noise=pyddm.NoiseConstant(noise=1), bound=self.bound,
                                 overlay=self.overlay, IC=self.IC, T_dur=self.T_dur)


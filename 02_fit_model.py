import pyddm
from pyddm import LossRobustBIC
import pyddm.plot
import models
import pandas as pd
import os
import utils
import matplotlib.pyplot as plt
import numpy as np

#Filter data, remove unrealistic response times and missing values
T_dur = 4.0
T_min = 0.5
df_rt = pd.read_csv("processed/measures/measures_vel.csv", sep=';')
df_rt = df_rt[(df_rt.Response_Time < T_dur) & (df_rt.Response_Time > T_min)]
df_rt = df_rt[~df_rt['Decision'].isin(['not finish overtake', 'calculation error', 'simulation error'])]

#Convert decision outcome to boolean
df_rt.Decision = df_rt.Decision.map({'Accepted gap': True, 'Rejected gap': False})
df_rt = df_rt[df_rt['a_mean'].notnull()]

#Create initial velocity array
v_start_data = pd.DataFrame({'v_start': df_rt['v_start']})
v_start_data = v_start_data.to_numpy()

#Create terciles to cluster initial velocity
terciles = df_rt["v_start"].quantile([0.33, 0.66])


def assign_tercile_label(x):
  if x <= terciles[0.33]:
      return "T1"
  elif x <= terciles[0.66]:
      return "T2"
  else:
      return "T3"
df_rt["v_start_cluster"] = df_rt["v_start"].apply(assign_tercile_label)
tercile_means = df_rt.groupby("v_start_cluster")["v_start"].mean()
df_rt["v_start_cluster"].replace({"T1": tercile_means[0], "T2": tercile_means[1], "T3": tercile_means[2]}, inplace=True)


#Load training data
overtaking_sample = pyddm.Sample.from_pandas_dataframe(df_rt, rt_column_name="Response_Time",
                                                       correct_column_name="Decision")
def fit_model(model):
  fitted_model = pyddm.fit_adjust_model(sample=overtaking_sample, model=model, lossfunction=LossRobustBIC,
                                        verbose=True)
  return fitted_model

def fit_model_by_condition():
  model_class = [models.ModelOutOfTheBox(),models.ModelOutOfTheBox_Bias(), models.ModelTTADistVel(), models.ModelTTADistVel_Bias(), models.Model_Drift_in_Bound(), models.Model_Drift_in_Bound_Bias(), models.Model_Vel_Drift_in_Bound(), models.Model_Vel_Drift_in_Bound_Bias() ]
  training_data = overtaking_sample
  print("len(training_data): " + str(len(training_data)))

  output_directory = "modeling"

  for model in model_class:
      subj_id = "all"  # instantiate the model
      file_name = "subj_parameters_fitted_" + model.__class__.__name__ + ".csv"
      if not os.path.isfile(os.path.join(output_directory, file_name)):
          utils.write_to_csv(output_directory, file_name, ["subj_id", "loss"] + model.param_names, write_mode="w")
      print(subj_id)

      fitted_model = fit_model(model.model)
      utils.write_to_csv(output_directory, file_name,
                     [subj_id, fitted_model.get_fit_result().value()]
                     + [float(param) for param in fitted_model.get_model_parameters()])

      #Plot fit of RT distribution
      pyddm.plot.plot_fit_diagnostics(fitted_model, sample=overtaking_sample)
      plt.savefig("model_fit_" + model.__class__.__name__ + ".png")
      plt.show()
  return fitted_model


fit_model_by_condition()



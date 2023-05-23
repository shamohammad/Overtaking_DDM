from scipy.optimize import minimize
import glob
import pandas
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


dt = 0.01  # time step for interpolation
t = np.arange(0, 4+dt, dt)


# Create an empty dataframe with the desired columns
columns = ['part_id', 'trial_no', 'run_no', 'time', 'condition_dist', 'v_start_cluster', 'd_ego_opposite', 'tta']
df_final = pandas.DataFrame(columns=columns)


# Loop over CSV files
for file in glob.glob(f'filtered_data/*.csv'):
   df = pandas.read_csv(file, sep=';')


   if 'd_ego_opposite' in df.columns and 'time' in df.columns:
       # Drop any rows with missing values
       df = df.dropna()
       df.replace([np.inf, -np.inf], np.nan, inplace=True)
       # Extract the relevant columns
       df = df.loc[:, ['part_id', 'run_no', 'trial_no', 'time', 'condition_distance', 'v_start_cluster','d_ego_opposite', 'tta']]


       # Calculate the difference between consecutive distance values
       diff = np.abs(np.diff(df['d_ego_opposite']))
       df['d_ego_opposite'] = np.abs(df['d_ego_opposite'])
       # Find the index of the first row where the distance changes by a certain threshold
       start_idx = np.argmax(diff > 50) + 1


       # Limit the data to a 4-second time range starting from the identified start index
       start_time = df.loc[start_idx, 'time']
       end_time = start_time + 4
       df = df.loc[(df['time'] >= start_time) & (df['time'] < end_time), :]


       # Shift the time values by subtracting the start time
       df['time'] -= start_time


       # Interpolate the distance and acc values onto the fixed time array
       f_d = interp1d(df['time'], df['d_ego_opposite'], kind='nearest', bounds_error=False, fill_value='extrapolate')
       f_tta = interp1d(df['time'], df['tta'], kind='nearest', bounds_error=False, fill_value='extrapolate')
       d_interp = f_d(t)
       tta_interp = f_tta(t)


       # Create a temporary dataframe with the interpolated values and the relevant columns
       df_temp = pandas.DataFrame({
           'part_id': np.tile(df[['part_id']].iloc[0], len(t)),
           'trial_no': np.tile(df[['trial_no']].iloc[0], len(t)),
           'run_no': np.tile(df[['run_no']].iloc[0], len(t)),
           'time': t,
           'condition_dist': np.tile(int(df[['condition_distance']].iloc[0]), len(t)),
           'v_start_cluster': np.tile(round(df[['v_start_cluster']].iloc[0],2), len(t)),
           'd_ego_opposite': d_interp,
           'tta': tta_interp
       })

       # Append the temporary dataframe to the final dataframe
       df_final = df_final.append(df_temp, ignore_index=True)

# Define the distance formula
def distance_formula(t, v_start_cluster, a_ego, a_opp, t_crit, v_opp):
   return v_start_cluster * t + 0.5 * a_ego * t**2 + 0.5 * a_opp * (t-t_crit)**2 + v_opp * (t-t_crit)


# Define the loss function to minimize
def loss_function(params, t, condition_dist, v_start_cluster, distance):
   a_ego, a_opp, t_crit, v_opp = params
   predicted_distance = condition_dist - distance_formula(t, v_start_cluster, a_ego, a_opp, t_crit, v_opp)
   return np.sum((predicted_distance[1:] - distance[1:]) ** 2)


# Extract the necessary columns from the DataFrame
condition = df_final['condition_dist'].values
v_start_cluster = df_final['v_start_cluster'].values
distance = df_final['d_ego_opposite'].values
time = df_final['time'].values

# Define the initial parameter values and bounds
initial_params = [0.125, 4, 1.2, 5] # starting values for a_ego, a_opp, t_crit, v_opp
param_bounds = [(0, 1), (0, 10), (0, 2), (0, 10)] # parameter bounds


# Define the initial distance for the condition
condition_dist = np.full(len(distance), np.mean(condition))

# Fit the model using the L-BFGS-B optimizer
result = minimize(loss_function, initial_params, args=(time, condition_dist, v_start_cluster, distance), bounds=param_bounds, method='L-BFGS-B')

# Extract the fitted parameter values
a_ego_fit, a_opp_fit, t_crit_fit, v_opp_fit = result.x

# Print the fitted parameter values
print("The fitted distance parameters are:")
print("a_ego = ", a_ego_fit)
print("a_opp = ", a_opp_fit)
print("t_crit = ", t_crit_fit)
print("v_opp = ", v_opp_fit)


# Group the data by time, condition_dist, and v_start_cluster and take the mean
df_grouped = df_final.groupby(['time', 'condition_dist', 'v_start_cluster']).mean()

# Get the unique values of condition_dist and v_start_cluster
condition_dists = df_final['condition_dist'].unique()
v_start_clusters = df_final['v_start_cluster'].unique()

t_actual = np.arange(401)* 0.01

fig, axes = plt.subplots(len(condition_dists), len(v_start_clusters), figsize=(12, 8), sharex=True, sharey=True)
for i, condition_dist in enumerate(condition_dists):
    for j, v_start_cluster in enumerate(v_start_clusters):
        data = df_grouped.loc[(t_actual, condition_dist, v_start_cluster), :]
        distance_actual = data["d_ego_opposite"].values
        distance_predicted = condition_dist - v_start_cluster * t_actual - 0.5 * a_ego_fit * t_actual ** 2 - 0.5 * a_opp_fit * (
                t_actual - t_crit_fit) ** 2 - v_opp_fit * (t_actual - t_crit_fit)
        ax = axes[i, j]
        ax.plot(t_actual, distance_actual, label="Actual")
        ax.plot(t_actual, distance_predicted, label="Predicted")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title(f"D0 = {condition_dist} m, v0_cluster = {v_start_cluster} m/s")
        ax.legend()
plt.tight_layout()
plt.show()

#Now for TTA
def tta_formula_num(t, v_start_cluster, a_ego, a_opp, t_crit):
   a_opp = np.where(t < t_crit, 0, a_opp)
   return v_start_cluster * t + 0.5 * a_ego * t ** 2 + 0.5 * a_opp * (t-t_crit)**2

def tta_formula_denum(t, v_start_cluster,a_ego, a_opp, t_crit):
   a_opp = np.where(t < t_crit, 0, a_opp)
   return v_start_cluster + a_ego * t + a_opp * (t-t_crit) + 0.0001

# Define the loss function to minimize
def loss_function_tta(params, t, condition_dist, v_start_cluster, tta):
  a_ego, a_opp, t_crit = params
  predicted_tta = (condition_dist - tta_formula_num(t, v_start_cluster, a_ego, a_opp, t_crit)) / tta_formula_denum(t, v_start_cluster, a_ego, a_opp, t_crit)
  return np.sum((predicted_tta[1:] - tta[1:]) ** 2)


# Extract the necessary columns from the DataFrame
condition = df_final['condition_dist'].values
v_start_cluster = df_final['v_start_cluster'].values
tta = df_final['tta'].values
time = df_final['time'].values

# Define the initial parameter values and bounds
initial_params = [0.125, 4, 1.2] # starting values for a_ego, a_opp, t_crit, v_opp
param_bounds = [(0, 1), (0, 10), (1.1, 1.3)] # parameter bounds

# Define the initial distance for the condition
condition_dist = np.full(len(tta), np.mean(condition))

# Fit the model using the L-BFGS-B optimizer
result_tta = minimize(loss_function_tta, initial_params, args=(time, condition_dist, v_start_cluster, tta), bounds=param_bounds, method='L-BFGS-B')

# Extract the fitted parameter values
a_ego_fit_tta, a_opp_fit_tta, t_crit_fit_tta = result_tta.x

# Group the data by time, condition_dist, and v_start_cluster and take the mean
df_grouped = df_final.groupby(['time', 'condition_dist', 'v_start_cluster']).mean()

# Get the unique values of condition_dist and v_start_cluster
condition_dists = df_final['condition_dist'].unique()
v_start_clusters = df_final['v_start_cluster'].unique()

t_actual = np.arange(401)* 0.01
distance_predicted= np.zeros_like(t_actual)
tta_predicted= np.zeros_like(t_actual)
fig, axes = plt.subplots(len(condition_dists), len(v_start_clusters), figsize=(12, 8), sharex=True, sharey=True)
for i, condition_dist in enumerate(condition_dists):
    for j, v_start_cluster in enumerate(v_start_clusters):
        data = df_grouped.loc[(t_actual, condition_dist, v_start_cluster), :]
        tta_actual = data["tta"].values
        a_opp_fit_tta = np.where(t_actual < t_crit_fit_tta, 0, a_opp_fit_tta)
        distance_predicted = condition_dist - v_start_cluster * t_actual - 0.5 * a_ego_fit * t_actual ** 2 - 0.5 * a_opp_fit * (
                t_actual - t_crit_fit) ** 2 - v_opp_fit * (t_actual - t_crit_fit)
        tta_predicted = distance_predicted / (v_start_cluster + a_ego_fit_tta * t_actual + a_opp_fit_tta * (t_actual-t_crit_fit_tta) + 0.0001)
        ax = axes[i, j]
        ax.plot(t_actual, tta_actual, label="Actual")
        ax.plot(t_actual, tta_predicted, label="Predicted")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("TTA (s)")
        ax.set_title(f"D0 = {condition_dist} m, v0_cluster = {v_start_cluster} m/s")
        ax.legend()
plt.tight_layout()
plt.show()

print("The fitted TTA parameters are:")
print("a_ego_tta = ", a_ego_fit_tta)
print("a_opp_tta = ", a_opp_fit_tta[-1])
print("t_crit_tta = ", t_crit_fit_tta)











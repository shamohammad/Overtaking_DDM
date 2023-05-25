# Overtaking_DDM

Built on Python 3.7

Step 0:
Download experiment_data_processed.zip from  https://osf.io/k3cmn/ and extract to folder where the python functions are located.

Step 1:
Create a virtual environment and install the packages lited in 'requirements.txt'.

Step 2: 
If you want to go through the whole process: proceed with Step 3.
If you want to directly obtain the plots depicted in the paper: proceed with Step 7.

Step 3:
Run '00_preprocess.py' to add clustered initial velocity to the respective participant data. Make sure you have the rights to modify files.

Step 4:
Run '01_kinematic_parameters.py' to model the distance and TTA functions. The output of this function are the fitted parameters for the kinematic equations.
By using kinematic equations instead of reading the actual mean distance and TTA the drift-diffusion model fitting is sped up significantly.

Step 5:
Run '02_fit_model.py' to initiate the model fitting. This part could take a couple of hours. It took in total about 4 hours on a i-9-9900K CPU @ 3.60GHz.

Step 6:
Run '03_simulation_models.ipynb' to simulate a range of initial velocity conditions in combination with the two distance conditions (160 m and 220 m).

Step 7:
Run '04_figures.ipynb' to obtain the plots depicted in the paper. 
Enter 'own' when Step 3 to 6 have been followed and you would like to obtain the results yourself. 
Enter 'paper' to directly obtain the plots depicted in the paper. 

*Note that differential evolution is a stochastic process so if a model behaves unexpected it could be that the global optimum was not found. In that case it is best to again run the model fitting process.

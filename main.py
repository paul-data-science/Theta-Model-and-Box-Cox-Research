'''
Author: PAUL AGGARWAL
Task: Research M4 Data Forecasting with Theta and Box Cox Parameter Lambda
Function: Main
'''
import config # Contains all functions with global variables!
import numpy as np

from scipy.optimize import minimize as om
from datetime import datetime
import pandas as pd

# leave reset_option ON incase someone sets display to max_rows!
pd.reset_option("max_rows")
pd.options.display.float_format = '{:,.5f}'.format
pd.set_option('display.max_columns', None)

start_time = datetime.now()
print("START Proc_Lambda_On_M4data_By_Each_Series:", datetime.now())

config.Proc_Lambda_On_M4data_By_Each_Series()
Lambda_On_M4data_By_Each_Series_df = pd.DataFrame(config.forecast_err_scores).T.reset_index()
Lambda_On_M4data_By_Each_Series_df = Lambda_On_M4data_By_Each_Series_df.infer_objects()

print("END Proc_Lambda_On_M4data_By_Each_Series:", datetime.now(), datetime.now()-start_time)

# Save to csv
print('Saving to CSV')
try:
    Lambda_On_M4data_By_Each_Series_df.to_csv('Lambda_On_M4data_By_Each_Series_df.csv')
    print('SUCCESS: Lambda_On_M4data_By_Each_Series_df.csv')
except:
    print('Something went wrong writing to csv! Doing next task....')
    
start_time = datetime.now()
print("START Each_Lambda_On_All_M4data:", datetime.now())

# Binary_Search_Opt_Rel_Owa_Lambda calls Proc_Each_Lambda_On_All_M4data
config.Binary_Search_Opt_Rel_Owa_Lambda(config.left_idx, config.right_idx)
Each_Lambda_On_All_M4data_df = pd.DataFrame(config.forecast_err_scores).T.reset_index()
Each_Lambda_On_All_M4data_df = Each_Lambda_On_All_M4data_df.infer_objects()

print("END Each_Lambda_On_All_M4data:", datetime.now(), datetime.now()-start_time)

# Save to csv
print('Saving to CSV')
try:
    Each_Lambda_On_All_M4data_df.to_csv('Each_Lambda_On_All_M4data_df.csv')
    print('SUCCESS: Each_Lambda_On_All_M4data_df.csv')
except:
    print('Something went wrong writing to csv! Doing next task....')

start_time = datetime.now()
print("START Proc_Lambda_On_M4data_By_Each_Series_Use_SciPy:", datetime.now())

config.Proc_Lambda_On_M4data_By_Each_Series_Use_SciPy()
Lambda_On_M4data_By_Each_Series_Use_SciPy_df = pd.DataFrame(config.forecast_err_scores).T.reset_index()
Lambda_On_M4data_By_Each_Series_Use_SciPy_df = Lambda_On_M4data_By_Each_Series_Use_SciPy_df.infer_objects()

print("END Proc_Lambda_On_M4data_By_Each_Series_Use_SciPy:", datetime.now(), datetime.now()-start_time)

# Save to csv
print('Saving to CSV')
try:
    Lambda_On_M4data_By_Each_Series_Use_SciPy_df.to_csv('Lambda_On_M4data_By_Each_Series_Use_SciPy_df.csv')
    print('SUCCESS: Lambda_On_M4data_By_Each_Series_Use_SciPy_df.csv')
except:
    print('Something went wrong writing to csv! Doing next task....')

start_time = datetime.now()
print("START Each_Lambda_On_All_M4data_Use_SciPy:", datetime.now())

# SciPy Optimize Minize module calls Proc_Each_Lambda_On_All_M4data_Use_SciPy
res = om(config.Proc_Each_Lambda_On_All_M4data_Use_SciPy, x0=0.5, bounds=[(0,1)])
Each_Lambda_On_All_M4data_Use_SciPy_df = pd.DataFrame(config.forecast_err_scores).T.reset_index()
Each_Lambda_On_All_M4data_Use_SciPy_df = Each_Lambda_On_All_M4data_Use_SciPy_df.infer_objects()

print("END Each_Lambda_On_All_M4data_Use_SciPy:", datetime.now(), datetime.now()-start_time)

# Save to csv
print('Saving to CSV')
try:
    Each_Lambda_On_All_M4data_Use_SciPy_df.to_csv('Each_Lambda_On_All_M4data_Use_SciPy_df.csv')
    print('SUCCESS: Each_Lambda_On_All_M4data_Use_SciPy_df.csv')
except:
    print('Something went wrong writing to csv!')

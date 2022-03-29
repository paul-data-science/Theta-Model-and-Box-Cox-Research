'''
Author: PAUL AGGARWAL
Task: Research M4 Data Forecasting with Theta and Box Cox Parameter Lambda
Function: Contains all Global Variables and Functions
'''

import numpy as np

# Initialize global array of lambda
lam_arr = np.arange(0, 1.1, 0.1)

# Initialize local variables for Binary Search
left_idx = 0
right_idx = len(lam_arr)-1

# M4 data Info CSV and files path locs
# Set as Global Var
m4_file_info_csv = 'C:/M4data/m4_info.csv'
m4_file_train_path = 'C:/M4data/'
m4_file_test_path = 'C:/M4data/'

def box_cox_transform(data, lam):
    import numpy as np
    
    if lam != 0.0:
        a = np.power(data, lam) - 1
        b = lam
        return a / b
    else:
        return np.log(data)

def box_cox_inverse(x_lam, lam):
    import numpy as np
    
    if lam == 0.0:
        return np.exp(x_lam)
    else:
        # there were some instances where final forecasted points were 
        # below zero so reversing it back created nan values
        
        inverse_transform = (lam * x_lam + 1)**(1/lam)
        return inverse_transform
        
        # Alternate formula
        #return np.exp(np.log(lam * x_lam + 1) / lam)
        

def simple_theta_method(train_data, horizon=1, alpha=0.5, y0=0, w=0.5):
    import numpy as np
    
    h = horizon
    y = np.asarray(train_data)
    
    # This was getting 'overflow encountered
    # in long_scalars' calc b1, used float64 to fix
    y_size = np.size(y)
    y_size = np.float64(y_size)
    x = np.arange(y_size)
   
    # X of h steps
    x_h = np.arange(y_size, y_size+h)
    
    a1 = y_size * np.sum(x * y)
    a2 = np.sum(x) * np.sum(y)
    b = np.sum(x * x)
    b1 = b * y_size
    b2 = np.sum(x)**2
    
    slope = (a1 - a2) / (b1 - b2)
    
    intercept = np.mean(y) - slope * np.mean(x)
    
    x_regr_line_theta_0 = intercept + slope * x
    
    # Compute Regression Line along X of h steps. 
    # This is forecasted regression line.
    x_h_regr_line_theta_0 = intercept + slope * x_h
    
    line_theta_2 = np.multiply(2, y) - x_regr_line_theta_0
    
    # Simple Exponential Smoothing over line_theta_2
    ses_theta_2_h = np.empty(h)
    ses_theta_2_before_h = np.empty(len(line_theta_2))

    ses_theta_2_before_h[0] = y0
    ses_theta_2_before_h[1] = line_theta_2[0]
    
    for t in range(2, np.size(ses_theta_2_before_h)):
        ses_theta_2_before_h[t] = alpha * line_theta_2[t-1] + (1-alpha) * ses_theta_2_before_h[t-1]
    
    ses_theta_2_h[0] = alpha * line_theta_2[t] + (1-alpha) * ses_theta_2_before_h[t]
    
    for t_h in range(1, np.size(ses_theta_2_h)):
        ses_theta_2_h[t_h] = ses_theta_2_h[t_h-1]
        
    # Finally, theta method forecast
    theta_0 = np.hstack((x_regr_line_theta_0, x_h_regr_line_theta_0))
    
    theta_2 = np.hstack((ses_theta_2_before_h, ses_theta_2_h))
   
    theta_forecast =  w * theta_0 + w * theta_2
    
    return theta_forecast[-h:]

def owa(smape_model, smape_naive2, mase_model, mase_naive2):
    import numpy as np

    try:
        smape_avg = smape_model / smape_naive2
    except:
        smape_avg = float('nan')
    
    try:
        mase_avg = mase_model / mase_naive2
    except:
        mase_avg = float('nan')

    return 0.5 * (smape_avg + mase_avg)

def mase(data, h, p, forecast):
    # formula referred to https://arxiv.org/pdf/2005.08067.pdf
    import numpy as np
    import math
    
    # Actual
    y = np.asarray(data)
    
    y_hat = np.asarray(forecast)
        
    m = p
    n = np.size(y)
    H = np.arange(n-h, n, 1)
    
    hi = np.arange(0, h, 1)
    
    # array index starts at zero not 1
    t = np.arange(0, n-h, 1) 
    
    # array index starts at period m, not 
    # m+1 because data index starts at zero
    j = np.arange(m, np.size(t)+h, 1) 
    
    a1 = np.sum(np.abs(y[H]-y_hat[hi]))
    a2 = np.size(H)
    b2 = np.size(t)+h-m
    b1 = np.sum(np.abs(y[j]-y[j-m]))
    if b1 == 0:
        # avoid divisible by zero. try/except doesn't work for numpy
        
        return float('nan')
        
        # Since using 'nansum' which turns NaN into zero
        # return 0 instead of 'nan'
        #return 0
    else:
        numer =  a1 / a2 
        denom = b1 / b2
        mase_ = numer / denom
        if math.isinf(mase_):
            # mask MASE if inf
            mase_ = float('nan')
            #mase_ = 0
        return mase_

def smape(data, h, forecast):
    import numpy as np
    
    # actual
    y = np.asarray(data)
    
    n = np.size(y)
    H = np.arange(n-h, n, 1)
    
    hi = np.arange(0, h, 1)
    
    y_hat = np.asarray(forecast)
    
    a1 = np.abs(y[H] - y_hat[hi])
    
    b1 = np.abs(y[H]) + np.abs(y_hat[hi])

    return 200/h * np.sum( a1 / b1 ) 

def naive2(train_data, h):
    import numpy as np
    # Equation Ref: Appendix A, https://www.nrpa.org/globalassets/journals/jlr/2003/volume-35/jlr-volume-35-number-4-pp-441-454.pdf
    # An Evaluation of Alternative Forecasting Methods to Recreation Visitation by CHEN, BLOOMFIELD AND FU
    # Naive 2 eq: A[t-1]*(1+((A[t-1] - A[t-2])/A[t-2])
    n = np.size(train_data)
    H = n + h
    y = train_data
    yhat = np.empty(H)
    yhat[:n+1] = np.nan
    yhat[n:] = y[n-1] * ( 1 + ( y[n-1] - y[n-2] ) / y[n-2] )
   
    return yhat[-h:]

def no_transformation_forecast(idx, train_data, test_data, h, p):
    global forecast_err_scores
    
    naive2_forecast_data = naive2(train_data, h)
    naive2_smape = smape(test_data, h, naive2_forecast_data)
    
    forecast_err_scores[idx]['Naive2_sMAPE'] = naive2_smape
    
    naive2_mase = mase(test_data, h, p, naive2_forecast_data)

    forecast_err_scores[idx]['Naive2_MASE'] = naive2_mase
    
    theta_forecast_data  = simple_theta_method(train_data, h, alpha=0.5, y0=0, w=0.5)
    theta_smape = smape(test_data, h, theta_forecast_data)
    
    forecast_err_scores[idx]['Theta_sMAPE'] = theta_smape
    
    theta_mase = mase(test_data, h, p, theta_forecast_data)
    
    forecast_err_scores[idx]['Theta_MASE'] = theta_mase
    
    theta_owa = owa(theta_smape, naive2_smape, theta_mase, naive2_mase)
    
    forecast_err_scores[idx]['Theta_OWA'] = theta_owa
    
def owa_lambda(lam, *args):
    global forecast_err_scores
    
    """
    (1) Transform train data using Box Cox Transform with respect to passed Lambda value, lam
    (2) Do the Theta method forecast
    (3) Do the Inverse of the Theta Forecast
    (4) Process the M4 benchmarks SMAPE, MASE, OWA
    """

    lam = float(lam)
    train_data_bc = box_cox_transform(train_data, lam)

    theta_forecast_bc = \
        simple_theta_method(train_data_bc, h, alpha=0.5, y0=0, w=0.5)

    train_data_untransform = box_cox_inverse(theta_forecast_bc, lam)
    
    forecast_err_scores[idx]['Lambda'] = lam
    
    theta_bc_smape = smape(test_data, h, train_data_untransform)
    forecast_err_scores[idx]['Theta_BC_sMAPE'] = theta_bc_smape

    theta_bc_mase = mase(test_data, h, p, train_data_untransform)
    forecast_err_scores[idx]['Theta_BC_MASE'] = theta_bc_mase

    theta_bc_owa = owa(forecast_err_scores[idx]['Theta_BC_sMAPE'], forecast_err_scores[idx]['Naive2_sMAPE'], \
                       forecast_err_scores[idx] ['Theta_BC_MASE'], forecast_err_scores[idx]['Naive2_MASE'])
    forecast_err_scores[idx]['Theta_BC_OWA'] = theta_bc_owa
    
    # Needed for Binary_Search_Opt_Owa_Lambda_By_Series() and SCiPy Optimize Minimize module
    return theta_bc_owa

def compute_relative_smape_mase_owa(idx):

    """
    Ref: https://usermanual.wiki/Pdf/M4CompetitorsGuide.719015056/view
    An  example for  computing  the OWA is  presented  below  using  the  MASE  and  sMAPE  of the M3 Competition methods:
    Divide all Errors by that of Naïve 2 to obtain the Relative MASE and the Relative sMAPE
    Compute the OWA by averaging the Relative MASE and the Relatives MAPE
    Note that MASE and sMAPE are first estimated per series by averaging the error estimated per forecasting horizon and 
    then averaged again across the 3003 time seriesto compute their value for the whole dataset. 
    On the other hand, OWA is computed only once at the end for the whole sample
    """
    # Global variables to keep internal state without OOP
    global sum_theta_bc_smape, sum_theta_bc_mase, sum_theta_smape, sum_theta_mase, sum_naive2_smape, sum_naive2_mase
    global forecast_err_scores
    
    if idx == 1:
        sum_theta_bc_smape = 0
        sum_theta_bc_mase = 0
        sum_theta_smape = 0
        sum_theta_mase = 0
        sum_naive2_smape = 0
        sum_naive2_mase = 0

        rel_theta_bc_smape = 0
        rel_theta_bc_mase = 0
        rel_theta_smape = 0
        rel_theta_mase = 0
        rel_naive2_smape = 0
        rel_naive2_mase = 0

        rel_theta_bc_owa = 0
        rel_theta_owa = 0
        
    
    sum_naive2_smape = np.nansum([sum_naive2_smape, forecast_err_scores[idx]['Naive2_sMAPE']])
    rel_naive2_smape = sum_naive2_smape/idx
    forecast_err_scores[idx]['Rel_Naive2_sMAPE'] = rel_naive2_smape
    
    sum_naive2_mase = np.nansum([sum_naive2_mase, forecast_err_scores[idx]['Naive2_MASE']])
    rel_naive2_mase = sum_naive2_mase/idx
    forecast_err_scores[idx]['Rel_Naive2_MASE'] = rel_naive2_mase
    
    sum_theta_smape = np.nansum([sum_theta_smape, forecast_err_scores[idx]['Theta_sMAPE']])
    rel_theta_smape = sum_theta_smape/idx
    forecast_err_scores[idx]['Rel_Theta_sMAPE'] = rel_theta_smape
    
    sum_theta_mase = np.nansum([sum_theta_mase, forecast_err_scores[idx]['Theta_MASE']])
    rel_theta_mase = sum_theta_mase/idx
    forecast_err_scores[idx]['Rel_Theta_MASE'] = rel_theta_mase
    
    rel_theta_owa = owa(rel_theta_smape, rel_naive2_smape, rel_theta_mase, rel_naive2_mase)
    forecast_err_scores[idx]['Rel_Theta_OWA'] = rel_theta_owa
    
    sum_theta_bc_smape = np.nansum([sum_theta_bc_smape, forecast_err_scores[idx]['Theta_BC_sMAPE']])
    rel_theta_bc_smape = sum_theta_bc_smape/idx
    forecast_err_scores[idx]['Rel_Theta_BC_sMAPE'] = rel_theta_bc_smape
    
    sum_theta_bc_mase = np.nansum([sum_theta_bc_mase, forecast_err_scores[idx]['Theta_BC_MASE']])
    rel_theta_bc_mase = sum_theta_bc_mase/idx
    forecast_err_scores[idx]['Rel_Theta_BC_MASE'] = rel_theta_bc_mase

    rel_theta_bc_owa = owa(rel_theta_bc_smape, rel_naive2_smape, rel_theta_bc_mase, rel_naive2_mase)
    forecast_err_scores[idx]['Rel_Theta_BC_OWA'] = rel_theta_bc_owa
    
    forecast_err_scores[idx]['Rel_Perc_OWA_Improvement'] = (rel_theta_owa-rel_theta_bc_owa)/rel_theta_owa*100

def Binary_Search_Opt_Rel_Owa_Lambda(left_idx, right_idx):
    '''
    This will process and search which lambda value made the lowest Theta Forecast OWA score
    on the entire M4 dataset files. 
    '''
    global lam_arr
    global idx
    
    # Initial middle, right owa scores
    mid_idx = (left_idx + right_idx) // 2
    Proc_Each_Lambda_On_All_M4data(lam_arr[mid_idx])
    mid_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']
    
    Proc_Each_Lambda_On_All_M4data(lam_arr[right_idx])
    right_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']

    while (left_idx < right_idx):
        
        if (mid_owa == right_owa):
            right_idx -= 1
            mid_idx = (left_idx + right_idx) // 2
            
            # Right OWA should be last dict process to get final output!   
            Proc_Each_Lambda_On_All_M4data(lam_arr[mid_idx])
            mid_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']
            
            Proc_Each_Lambda_On_All_M4data(lam_arr[right_idx])
            right_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']

        elif (mid_owa > right_owa):
            left_idx = mid_idx + 1
            mid_idx = (left_idx + right_idx) // 2

            Proc_Each_Lambda_On_All_M4data(lam_arr[mid_idx])
            mid_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']
            
        else:
            right_idx = mid_idx
            mid_idx = (left_idx + right_idx) // 2
            
            # Right OWA should be last dict process to get final output!
            Proc_Each_Lambda_On_All_M4data(lam_arr[mid_idx])
            mid_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']
            
            Proc_Each_Lambda_On_All_M4data(lam_arr[right_idx])
            right_owa = forecast_err_scores[idx]['Rel_Theta_BC_OWA']

def Binary_Search_Opt_Owa_Lambda_By_Series(left_idx, right_idx, train_data, test_data, h, p, naive2_smape, naive2_mase):
    global lam_arr
    
    # Initial middle, right owa scores
    mid_idx = (left_idx + right_idx) // 2

    mid_owa = owa_lambda(lam_arr[mid_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)
    right_owa = owa_lambda(lam_arr[right_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)

    while (left_idx < right_idx):
        
        if (mid_owa == right_owa):
            right_idx -= 1
            mid_idx = (left_idx + right_idx) // 2
            
            # Right OWA should be last dict process to get final output!
            mid_owa = owa_lambda(lam_arr[mid_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)
            right_owa = owa_lambda(lam_arr[right_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)

        elif (mid_owa > right_owa):
            left_idx = mid_idx + 1
            mid_idx = (left_idx + right_idx) // 2

            mid_owa = owa_lambda(lam_arr[mid_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)
          
        else:
            right_idx = mid_idx
            mid_idx = (left_idx + right_idx) // 2
            
            # Right OWA should be last dict process to get final output!
            mid_owa = owa_lambda(lam_arr[mid_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)
            right_owa = owa_lambda(lam_arr[right_idx], train_data, test_data, h, p, naive2_smape, naive2_mase)
            

def Proc_Lambda_On_M4data_By_Each_Series():
    import pandas as pd
    import csv
    import numpy as np
   
    """
    Call this function first to get Optimal Lambda for each M4 series.
    This Porcess uses Binary_Search_Opt_Owa_Lambda_By_Series.
    
    """
       
    global idx, forecast_err_scores
    global train_data, test_data
    global h, p
    global m4_file_info_csv, m4_file_train_path, m4_file_test_path
    
    forecast_err_scores = {}
    
    df_info = pd.read_csv(m4_file_info_csv)

    # Make list of m4 file names from M4 info csv
    file_list = df_info['SP'].unique().tolist()
    
    idx = 1
    forecast_err_scores[idx] = {}
    
    with open(m4_file_info_csv) as csv_info:
        csv_obj_info = csv.reader(csv_info)
        next(csv_obj_info) # goto header in M4 data file
        row_info = next(csv_obj_info) # goto first data row in M4 data file
        for filename in file_list:
            with open(m4_file_train_path+filename+"-train.csv") as csv_train, \
                        open(m4_file_test_path+filename+"-test.csv") as csv_test:
                csv_obj_train = csv.reader(csv_train)
                csv_obj_test = csv.reader(csv_test)

                next(csv_obj_train) # goto header in M4 data file
                next(csv_obj_test) # goto header in M4 data file

                row_test = next(csv_obj_test) # goto first data row in M4 data file

                row_cnt = 0 # keep track of file rows

                for row_train in csv_obj_train:

                    row_cnt += 1

                    # Check for matching test series
                    if (row_test[0] != row_train[0]) or (row_info[0] != row_test[0]) or (row_info[0] != row_train[0]):
                        print('IDs between the files do not match: ', row_cnt, row_info[0], row_train[0], row_test[0])
                        break

                    row_train = list(filter(None, row_train))
                    row_test = list(filter(None, row_test))

                    try:
                        series_train = {'id': row_train[0], 'data': np.array(row_train[1:]).astype(float) }
                        train_id = series_train['id']
                        train_data = series_train['data']
                    except:
                        print(series_train)
                        print('Train EOF')
                        break

                    try:
                        series_test = {'id': row_test[0], 'data': np.array(row_test[1:]).astype(float) }
                        test_id = series_test['id']
                        test_data = series_test['data']
                    except:
                        print(series_test)
                        print('Something wrong with Test parsing')
                        break

                    try:
                        series_info = {'id': row_info[0], 'cat': row_info[1], 'freq': int(row_info[2]),\
                                       'h': int(row_info[3]), 'sp': row_info[4],\
                                      'start_date': row_info[5]}
                        info_id = series_info['id']
                        cat = series_info['cat']
                        p = series_info['freq']
                        h = series_info['h']
                        sp = series_info['sp']
                        info_start_date = series_info['start_date']
                    except:
                        print(series_info)
                        print('Something wrong with Info parsing')
                        break
                    
                    # This condition was made when using M3 data set which did not have separate
                    # test set. We had to split train set by specific horizon to get test set.
                    '''
                    if (np.size(train_data) - 2) <= h:
                        print('Length of horizon time series is nearly same as training time series.')
                        print('Will skip this series ', train_id,\
                              ' for now, unless you change your horizon to be more than 2 points less than ',\
                              np.size(train_data), ' Train data.')
                    '''
                    else:

                        forecast_err_scores[idx]['Series_Id'] = info_id
                        forecast_err_scores[idx]['SP'] = sp
                        forecast_err_scores[idx]['Cat'] = cat
                        forecast_err_scores[idx]['Horizon'] = h
                        forecast_err_scores[idx]['Periods'] = p

                        # Process Naive2, Theta forecasts with sMAPE, MASE and OWA dict
                        no_transformation_forecast(idx, train_data, test_data, h, p)

                        Binary_Search_Opt_Owa_Lambda_By_Series(left_idx, right_idx,train_data, test_data, h, p, \
                                                    forecast_err_scores[idx]['Naive2_sMAPE'], forecast_err_scores[idx]['Naive2_MASE'])

                        compute_relative_smape_mase_owa(idx)


                        #if idx == 1:
                            #break
                        try:
                            row_test = next(csv_obj_test) # goto next data row in M4 data file
                        except:
                            print(row_test)
                            print('Test data EOF')

                        # row_info object needs to be last
                        # otherwise will not sync (step ahead) with train and test files on next filename loop!
                        try:
                            row_info = next(csv_obj_info) # goto next data row in M4 data file
                            idx += 1
                            forecast_err_scores[idx] = {}
                        except:
                            print(row_info)
                            print('Info data EOF')
                            
def Proc_Each_Lambda_On_All_M4data(lam):
    import csv
    import pandas as pd

    """
    Note:
    Use Binary_Search_Opt_Rel_Owa_Lambda to call this function!
    This will find common Optimal Lambda for all M4 data.
    
    """
    global idx, forecast_err_scores
    global train_data, test_data
    global h, p
    global m4_file_info_csv, m4_file_train_path, m4_file_test_path
    
    forecast_err_scores = {}
    
    df_info = pd.read_csv(m4_file_info_csv)

    # Make list of m4 file names from M4 Info csv
    file_list = df_info['SP'].unique().tolist()
    
    idx = 1
    forecast_err_scores[idx] = {}
    
    with open(m4_file_info_csv) as csv_info:
        csv_obj_info = csv.reader(csv_info)
        next(csv_obj_info) # goto header in M4 data file
        row_info = next(csv_obj_info) # goto first data row in M4 data file
        for filename in file_list:
            with open(m4_file_train_path+filename+"-train.csv") as csv_train, \
                        open(m4_file_test_path+filename+"-test.csv") as csv_test:
                csv_obj_train = csv.reader(csv_train)
                csv_obj_test = csv.reader(csv_test)

                next(csv_obj_train) # goto header in M4 data file
                next(csv_obj_test) # goto header in M4 data file

                row_test = next(csv_obj_test) # goto first data row in M4 data file

                row_cnt = 0 # keep track of file rows

                for row_train in csv_obj_train:

                    row_cnt += 1

                    # Check for matching test series
                    if (row_test[0] != row_train[0]) or (row_info[0] != row_test[0]) or (row_info[0] != row_train[0]):
                        print('IDs between the files do not match: ', row_cnt, row_info[0], row_train[0], row_test[0])
                        break

                    row_train = list(filter(None, row_train))
                    row_test = list(filter(None, row_test))

                    try:
                        series_train = {'id': row_train[0], 'data': np.array(row_train[1:]).astype(float) }
                        train_id = series_train['id']
                        train_data = series_train['data']
                    except:
                        print(series_train)
                        print('Train EOF')
                        break

                    try:
                        series_test = {'id': row_test[0], 'data': np.array(row_test[1:]).astype(float) }
                        test_id = series_test['id']
                        test_data = series_test['data']
                    except:
                        print(series_test)
                        print('Something wrong with Test parsing')
                        break

                    try:
                        series_info = {'id': row_info[0], 'cat': row_info[1], 'freq': int(row_info[2]),\
                                       'h': int(row_info[3]), 'sp': row_info[4],\
                                      'start_date': row_info[5]}
                        info_id = series_info['id']
                        cat = series_info['cat']
                        p = series_info['freq']
                        h = series_info['h']
                        sp = series_info['sp']
                        info_start_date = series_info['start_date']
                    except:
                        print(series_info)
                        print('Something wrong with Info parsing')
                        break
                    
                    # This condition was made when using M3 data set which did not have separate
                    # test set. We had to split train set by specific horizon to get test set.
                    '''
                    if (np.size(train_data) - 2) <= h:
                        print('Length of horizon time series is nearly same as training time series.')
                        print('Will skip this series ', train_id,\
                              ' for now, unless you change your horizon to be more than 2 points less than ',\
                              np.size(train_data), ' Train data.')
                    '''
                    else:

                        forecast_err_scores[idx]['Series_Id'] = info_id
                        forecast_err_scores[idx]['SP'] = sp
                        forecast_err_scores[idx]['Cat'] = cat
                        forecast_err_scores[idx]['Horizon'] = h
                        forecast_err_scores[idx]['Periods'] = p

                        # Process Naive2, Theta forecasts with sMAPE, MASE and OWA dict
                        no_transformation_forecast(idx, train_data, test_data, h, p)
                        owa_lambda(lam, idx, train_data, test_data, h, p, forecast_err_scores[idx]['Naive2_sMAPE'], forecast_err_scores[idx]['Naive2_MASE'] )
                        compute_relative_smape_mase_owa(idx)

                        #if idx == 1:
                            #break
                        try:
                            row_test = next(csv_obj_test) # goto next data row in M4 data file
                        except:
                            print(row_test)
                            print('Test data EOF')

                        # row_info object needs to be last
                        # otherwise will not sync (step ahead) with train and test files on next filename loop!
                        try:
                            row_info = next(csv_obj_info) # goto next data row in M4 data file
                            idx += 1
                            forecast_err_scores[idx] = {}
                        except:
                            print(row_info)
                            print('Info data EOF')
                            
def Proc_Lambda_On_M4data_By_Each_Series_Use_SciPy():
    from scipy.optimize import minimize as om
    import pandas as pd
    import csv
    
    """
    Call this function first to get Optimal Lambda for each M4 series.
    This Porcess uses scipy.optimize.minimize.
    
    """

    global idx, forecast_err_scores
    global train_data, test_data
    global h, p
    global m4_file_info_csv, m4_file_train_path, m4_file_test_path
    
    forecast_err_scores = {}
    
    df_info = pd.read_csv(m4_file_info_csv)

    # Make list of m4 file names from M4 info csv
    file_list = df_info['SP'].unique().tolist()
    
    idx = 1
    forecast_err_scores[idx] = {}
    
    with open(m4_file_info_csv) as csv_info:
        csv_obj_info = csv.reader(csv_info)
        next(csv_obj_info) # goto header in M4 data file
        row_info = next(csv_obj_info) # goto first data row in M4 data file
        for filename in file_list:
            with open(m4_file_train_path+filename+"-train.csv") as csv_train, \
                        open(m4_file_test_path+filename+"-test.csv") as csv_test:
                csv_obj_train = csv.reader(csv_train)
                csv_obj_test = csv.reader(csv_test)

                next(csv_obj_train) # goto header in M4 data file
                next(csv_obj_test) # goto header in M4 data file

                row_test = next(csv_obj_test) # goto first data row in M4 data file

                row_cnt = 0 # keep track of file rows

                for row_train in csv_obj_train:

                    row_cnt += 1

                    # Check for matching test series
                    if (row_test[0] != row_train[0]) or (row_info[0] != row_test[0]) or (row_info[0] != row_train[0]):
                        print('IDs between the files do not match: ', row_cnt, row_info[0], row_train[0], row_test[0])
                        break

                    row_train = list(filter(None, row_train))
                    row_test = list(filter(None, row_test))

                    try:
                        series_train = {'id': row_train[0], 'data': np.array(row_train[1:]).astype(float) }
                        train_id = series_train['id']
                        train_data = series_train['data']
                    except:
                        print(series_train)
                        print('Train EOF')
                        break

                    try:
                        series_test = {'id': row_test[0], 'data': np.array(row_test[1:]).astype(float) }
                        test_id = series_test['id']
                        test_data = series_test['data']
                    except:
                        print(series_test)
                        print('Something wrong with Test parsing')
                        break

                    try:
                        series_info = {'id': row_info[0], 'cat': row_info[1], 'freq': int(row_info[2]),\
                                       'h': int(row_info[3]), 'sp': row_info[4],\
                                      'start_date': row_info[5]}
                        info_id = series_info['id']
                        cat = series_info['cat']
                        p = series_info['freq']
                        h = series_info['h']
                        sp = series_info['sp']
                        info_start_date = series_info['start_date']
                    except:
                        print(series_info)
                        print('Something wrong with Info parsing')
                        break
                    
                    # This condition was made when using M3 data set which did not have separate
                    # test set. We had to split train set by specific horizon to get test set.
                    '''
                    if (np.size(train_data) - 2) == h:
                        print('Length of horizon time series is nearly same as training time series.')
                        print('Will skip this series ', train_id,\
                              ' for now, unless you change your horizon to be more than 2 points less than ',\
                              np.size(train_data), ' Train data.')
                    '''
                    
                    else:

                        forecast_err_scores[idx]['Series_Id'] = info_id
                        forecast_err_scores[idx]['SP'] = sp
                        forecast_err_scores[idx]['Cat'] = cat
                        forecast_err_scores[idx]['Horizon'] = h
                        forecast_err_scores[idx]['Periods'] = p

                        # Process Naive2, Theta forecasts with sMAPE, MASE and OWA dict
                        no_transformation_forecast(idx, train_data, test_data, h, p)

                        res = om(owa_lambda , x0=0.5 , args=(idx, info_id, train_id, train_data, test_data, h, p, \
                                                     forecast_err_scores[idx]['Naive2_sMAPE'], \
                                                     forecast_err_scores[idx]['Naive2_MASE'],), bounds=[(0,1)])
                        
                        compute_relative_smape_mase_owa(idx)


                        #if idx == 1:
                            #break
                        try:
                            row_test = next(csv_obj_test) # goto next data row in M4 data file
                        except:
                            print(row_test)
                            print('Test data EOF')

                        # row_info object needs to be last
                        # otherwise will not sync (step ahead) with train and test files on next filename loop!
                        try:
                            row_info = next(csv_obj_info) # goto next data row in M4 data file
                            idx += 1
                            forecast_err_scores[idx] = {}
                        except:
                            print(row_info)
                            print('Info data EOF')
                            
                            
def Proc_Each_Lambda_On_All_M4data_Use_SciPy(lam):
    import csv
    import pandas as pd
    
    """
    Note:
    Use Scipy Optimize Minimize function to call this function!
    This function will find common Optimal Lambda for all M4 data.
    
    """

    global idx, forecast_err_scores
    global train_data, test_data
    global h, p
    global m4_file_info_csv, m4_file_train_path, m4_file_test_path
    
    forecast_err_scores = {}
    
    df_info = pd.read_csv(m4_file_info_csv)

    # Make list of m4 file names from M4 Info csv
    file_list = df_info['SP'].unique().tolist()
    
    idx = 1
    forecast_err_scores[idx] = {}
    
    with open(m4_file_info_csv) as csv_info:
        csv_obj_info = csv.reader(csv_info)
        next(csv_obj_info) # goto header in M4 data file
        row_info = next(csv_obj_info) # goto first data row in M4 data file
        for filename in file_list:
            with open(m4_file_train_path+filename+"-train.csv") as csv_train, \
                        open(m4_file_test_path+filename+"-test.csv") as csv_test:
                csv_obj_train = csv.reader(csv_train)
                csv_obj_test = csv.reader(csv_test)

                next(csv_obj_train) # goto header in M4 data file
                next(csv_obj_test) # goto header in M4 data file

                row_test = next(csv_obj_test) # goto first data row in M4 data file

                row_cnt = 0 # keep track of file rows

                for row_train in csv_obj_train:

                    row_cnt += 1

                    # Check for matching test series
                    if (row_test[0] != row_train[0]) or (row_info[0] != row_test[0]) or (row_info[0] != row_train[0]):
                        print('IDs between the files do not match: ', row_cnt, row_info[0], row_train[0], row_test[0])
                        break

                    row_train = list(filter(None, row_train))
                    row_test = list(filter(None, row_test))

                    try:
                        series_train = {'id': row_train[0], 'data': np.array(row_train[1:]).astype(float) }
                        train_id = series_train['id']
                        train_data = series_train['data']
                    except:
                        print(series_train)
                        print('Train EOF')
                        break

                    try:
                        series_test = {'id': row_test[0], 'data': np.array(row_test[1:]).astype(float) }
                        test_id = series_test['id']
                        test_data = series_test['data']
                    except:
                        print(series_test)
                        print('Something wrong with Test parsing')
                        break

                    try:
                        series_info = {'id': row_info[0], 'cat': row_info[1], 'freq': int(row_info[2]),\
                                       'h': int(row_info[3]), 'sp': row_info[4],\
                                      'start_date': row_info[5]}
                        info_id = series_info['id']
                        cat = series_info['cat']
                        p = series_info['freq']
                        h = series_info['h']
                        sp = series_info['sp']
                        info_start_date = series_info['start_date']
                    except:
                        print(series_info)
                        print('Something wrong with Info parsing')
                        break

                    if (np.size(train_data) - 2) == h:
                        print('Length of horizon time series is nearly same as training time series.')
                        print('Will skip this series ', train_id,\
                              ' for now, unless you change your horizon to be 2 points less than ',\
                              np.size(train_data), ' Train data.')
                    else:

                        forecast_err_scores[idx]['Series_Id'] = info_id
                        forecast_err_scores[idx]['SP'] = sp
                        forecast_err_scores[idx]['Cat'] = cat
                        forecast_err_scores[idx]['Horizon'] = h
                        forecast_err_scores[idx]['Periods'] = p

                        # Process Naive2, Theta forecasts with sMAPE, MASE and OWA dict
                        no_transformation_forecast(idx, train_data, test_data, h, p)
                        
                        owa_lambda(lam, idx, train_data, test_data, h, p, forecast_err_scores[idx]['Naive2_sMAPE'], forecast_err_scores[idx]['Naive2_MASE'] )
                        
                        compute_relative_smape_mase_owa(idx)

                        #if idx == 1:
                            #break
                        try:
                            row_test = next(csv_obj_test) # goto next data row in M4 data file
                        except:
                            print(row_test)
                            print('Test data EOF')

                        # row_info object needs to be last
                        # otherwise will not sync (step ahead) with train and test files on next filename loop!
                        try:
                            row_info = next(csv_obj_info) # goto next data row in M4 data file
                            idx += 1
                            forecast_err_scores[idx] = {}
                        except:
                            print(row_info)
                            print('Info data EOF')
    
    # This is needed by the Scipy.Optimize.Minimize which called this function
    return forecast_err_scores[idx]['Rel_Theta_BC_OWA']

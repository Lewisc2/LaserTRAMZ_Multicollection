#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:18:40 2024

@author: ctlewis
"""


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.layouts import row, gridplot
from bokeh.models import Range1d
import panel as pn
import statistics
import param
import sys
import statsmodels.api as sm
from patsy import dmatrices
import scipy
from scipy import stats
from scipy.optimize import curve_fit
from itertools import cycle
import matplotlib as mpl
mpl.use('agg')
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

volt_count_constant = 1.602e-8 #volts / count
color_palette = bokeh.palettes.Muted9
color_palette_regressions = bokeh.palettes.Dark2_3

class calc_fncs:
    """ Class that holds all of the functions for reducing the time resolved data"""
    def __init__(self,*args):
        for a in args:
            self.__setattr__(str(a), args[0])
    
    def get_ratios(data):
        """
        function that calculates relevant isotopic ratios for the U-Pb decay system 235/238 is returned strictly to make 
        plotting easy.
        
        Parameters
        ----------
        data : pandas dataframe
            pandas dataframe holding the observed time resolved LAICPMS measurements
    
        Returns
        -------
        data_ratio : pandas dataframe
            pandas dataframe with calculated isotopic ratios of interest. Note these are note the Hg corrected ratios for 204.
            These are the ratios that are used specifically for plotting and visualizing the measured ratios.
        """
        
        og_len = len(data.columns) # get the length of columns
        data = data.reset_index(drop=True) # reset index to always set the data into a format that allows data to be manipulated
        data_ratio = data.copy() # create copy of df so that nothing is overwritten
        # initialize arrays to be filled with calculated ratios from the data
        pb206_u238 = []
        u238_pb206 = []
        pb207_u235 = []
        pb208_th232 = []
        pb207_pb206 = []
        u238_u235 = []
        u238_th232 = []
        pb206_pb204 = []
        pb207_pb204 = []
        
        pb206_u238 = np.zeros(len(data_ratio))
        u238_pb206 = np.zeros(len(data_ratio))
        pb207_u235 = np.zeros(len(data_ratio))
        pb208_th232 = np.zeros(len(data_ratio))
        pb207_pb206 = np.zeros(len(data_ratio))
        u238_u235 = np.zeros(len(data_ratio))
        u238_th232 = np.zeros(len(data_ratio))
        pb206_pb204 = np.zeros(len(data_ratio))
        pb207_pb204 = np.zeros(len(data_ratio))
        
        
        # create loops that append 1) calculated ratio for each observation if denominator > 0 or 
        # 2) zero in the case that denominator = 0 (avoids dividing by zero)
        for i in range(0,len(data_ratio)): # loop through range of the data
            if data_ratio['238U'][i] > 0 and data_ratio['206Pb'][i] > 0: # test for denominator > 0
                pb206_u238[i] = (data_ratio['206Pb'][i]/data_ratio['238U'][i]) # append ratio if condition satisfied
                u238_pb206[i] = (data_ratio['238U'][i]/data_ratio['206Pb'][i])
            else: # append zero otherwise
                pb206_u238[i] = 0
                u238_pb206[i] = 0
            
        for i in range(0,len(data_ratio)):
            if data_ratio['235U'][i] > 0 and data_ratio['207Pb'][i] > 0:
                pb207_u235[i] = (data_ratio['207Pb'][i]/data_ratio['235U'][i])
            else:
                pb207_u235[i] = 0
                
        for i in range(0,len(data_ratio)):
            if data_ratio['208Pb'][i] > 0 and data_ratio['232Th'][i] > 0:
                pb208_th232[i] = (data_ratio['208Pb'][i]/data_ratio['232Th'][i])
            else:
                pb208_th232[i] = 0
            
        for i in range(0,len(data_ratio)):
            if data_ratio['206Pb'][i] > 0 and data_ratio['207Pb'][i] > 0:
                pb207_pb206[i] = (data_ratio['207Pb'][i]/data_ratio['206Pb'][i])
            else:
                pb207_pb206[i] = 0
                    
                
        for i in range(0,len(data_ratio)):
            if data_ratio['235U'][i] > 0 and data_ratio['238U'][i] > 0:
                u238_u235[i] = (data_ratio['238U'][i]/data_ratio['235U'][i])
            else:
                u238_u235[i] = 0
        
        for i in range(0,len(data_ratio)):
            if data_ratio['238U'][i] > 0 and data_ratio['232Th'][i] > 0:
                u238_th232[i] = (data_ratio['238U'][i]/data_ratio['232Th'][i])
            else:
                u238_th232[i] = 0
                
        for i in range(0,len(data_ratio)):
            if data_ratio['206Pb'][i] > 0 and data_ratio['204Pb'][i] > 0:
                pb206_pb204[i] = (data_ratio['206Pb'][i]/data_ratio['204Pb'][i])
            else:
                pb206_pb204[i] = 0
                
        for i in range(0,len(data_ratio)):
            if data_ratio['207Pb'][i] > 0 and data_ratio['204Pb'][i] > 0:
                pb207_pb204[i] = (data_ratio['207Pb'][i]/data_ratio['204Pb'][i])
            else:
                pb207_pb204[i] = 0
                
        # insert the lists into the copied dataframe
        data_ratio['206Pb/238U'] = pb206_u238
        data_ratio['238U/206Pb'] = u238_pb206
        data_ratio['207Pb/235U'] = pb207_u235
        data_ratio['208Pb/232Th'] = pb208_th232
        data_ratio['207Pb/206Pb'] = pb207_pb206
        data_ratio['238U/235U'] = u238_u235
        data_ratio['238U/232Th'] = u238_th232
        data_ratio['206Pb/204Pb'] = pb206_pb204
        data_ratio['207Pb/204Pb'] = pb207_pb204
        
        data_ratio = data_ratio.iloc[:,(og_len-1):] # insert the calculated ratios onto the end of the copied dataframe
        
        return data_ratio

    
    def threesig_outlierremoval(data):
        whileloopdata = data.reset_index(drop=True)
        ratios = ['206Pb/238U','207Pb/235U','207Pb/206Pb','238U/235U','238U/206Pb','208Pb/232Th','238U/232Th','206Pb/204Pb']
        max_iter = 1000
        loopiter = 0
            
        for r in ratios:
            threesig = 3*whileloopdata[r].std()
            mean = whileloopdata[r].mean()
            loopvariable = True
            trigger = False
            while loopvariable == True:
                # break the loop if the regression fails and the array is nan of length 1
                if len(whileloopdata)<=2:
                    break
                elif loopiter >= max_iter:
                    print('Hit Max Iterations for despike')
                    break
                else:
                    for i in range(1,len(whileloopdata)-1):
                        # check if point is a 3sigma outlier
                        if np.abs(mean-whileloopdata.loc[i,r]) > mean+threesig:
                            whileloopdata.loc[i,r] = np.mean([whileloopdata.loc[i-1,r],whileloopdata.loc[i+1,r]]) # interpolate with mean of two nearest points
                            threesig = 3*whileloopdata[r].std() # recalculate three sigma with updated point
                            mean = whileloopdata[r].mean() # recalculate mean with updated point
                            trigger = True # fire trigger that prevents while variable to switch to false and reloop with updated data
                        else:
                            whileloopdata.loc[i,r] = whileloopdata.loc[i,r]
                    # if some datapoint was triggered as a threesig outlier, leave the var as true
                    # otherwise change the var to false to exit the while loop
                    if trigger == True:
                        trigger = False # reset trigger variable in prep for next loop
                    else:
                        loopvariable = False
                        print('Exited Loop After despike')
                loopiter = loopiter+1
                        
        return whileloopdata
    
    
    def get_counts(data,counts_mode,ablation_start_true,bckgrnd_start_input,bckgrnd_stop_input,ablation_start_input,ablation_stop_input, integration_time,analyte_cols):
        """
        Function that gets the ratio / counts data for the relevant ratios and analytes.

        Parameters
        ----------
        data : dataframe
            pandas dataframe holding the data from the current analysis.
        counts_mode : string
            string denoting which method the using wants to reduce data with.
        ablation_start_true : float
            float value of the projected ablation start time.
        bckgrnd_start_input : integer
            integer from the slider defining when to start taking the gas blank.
        bckgrnd_stop_input : integer
            integer from the slider defining when to stop taking the gas blank.
        ablation_start_input : integer
            integer from the slider defining when to start the regression / counts.
        ablation_stop_input : integer
            integer form the slider defining when to stop the regression / counts.
        integration_time: float
            float value used to calculate counts from cps
        analyte_cols: object
            object containing a list of analytes (in format MassElement, e.g., 238U)

        Returns
        -------
        ratios_to_return : array
            array of values including relevant ratios and analyte intensities when applicable.
        stats_to_return : array
            array of values including relevant errors and regression statistics.

        """
        
        if counts_mode == 'Total Counts':
            start_analyte = analyte_cols[0] # get first analyte in columns
            stop_analyte = analyte_cols[-1] # get last analyte in columns
            # get counts of all analytes in the dataframe by multiply each pass by integration time, then summing all observations
            data_totalcounts = np.sum(data.loc[:,start_analyte:stop_analyte] * integration_time)
            data_ratios_ = calc_fncs.get_ratios(data) # get ratios
            data_ratios_ = calc_fncs.threesig_outlierremoval(data_ratios_)
            pb206_u238 = data_totalcounts['206Pb']/data_totalcounts['238U']
            u238_pb206 = data_totalcounts['238U']/data_totalcounts['206Pb']
            pb207_u235 = data_totalcounts['207Pb']/data_totalcounts['235U']
            pb208_th232 = data_totalcounts['208Pb']/data_totalcounts['232Th']
            u238_235 = data_totalcounts['238U']/data_totalcounts['235U']
            u238_th232 = data_totalcounts['238U']/data_totalcounts['232Th']
            pb207_206 = data_totalcounts['207Pb']/data_totalcounts['206Pb']
            pb206_204 = data_totalcounts['206Pb']/data_totalcounts['204Pb']
            pb207_204 = data_totalcounts['207Pb']/data_totalcounts['204Pb']
            
            
            # community accepted SE of measurement on ratios and their percentages
            pb206_u238SE_percent = data_ratios_['206Pb/238U'].sem()/pb206_u238*100
            u238_pb206SE_percent = data_ratios_['238U/206Pb'].sem()/u238_pb206*100
            pb207_u235SE_percent = data_ratios_['207Pb/235U'].sem()/pb207_u235*100
            pb208_th232SE_percent = data_ratios_['208Pb/232Th'].sem()/pb208_th232*100
            u238_235SE_percent = data_ratios_['238U/235U'].sem()/u238_235*100
            u238_th232SE_percent = data_ratios_['238U/232Th'].sem()/u238_th232*100
            pb207_206SE_percent = data_ratios_['207Pb/206Pb'].sem()/pb207_206*100
            pb206_204SE_percent = data_ratios_['206Pb/204Pb'].sem()/pb206_204*100
            pb207_204SE_percent = data_ratios_['207Pb/204Pb'].sem()/pb207_204*100

            
            ratios_to_return = [pb206_u238,u238_pb206,pb207_u235,pb208_th232,pb207_206,u238_235,u238_th232,pb206_204,pb207_204]
            stats_to_return = [pb206_u238SE_percent,u238_pb206SE_percent,pb207_u235SE_percent,pb208_th232SE_percent,
                               pb207_206SE_percent,u238_235SE_percent,u238_th232SE_percent,
                               pb206_204SE_percent,pb207_204SE_percent]
            
            return ratios_to_return,stats_to_return
            
            
        elif counts_mode == 'Means & Regression':
            data_ratios_ = calc_fncs.get_ratios(data)
            data_ratios_ = calc_fncs.threesig_outlierremoval(data_ratios_)
            pb206_204 = data_ratios_['206Pb/204Pb'].mean()
            pb207_204 = data_ratios_['207Pb/204Pb'].mean()
            pb207_206 = data_ratios_['207Pb/206Pb'].mean()
            u238_235 = data_ratios_['238U/235U'].mean()
            u238_th232 = data_ratios_['238U/232Th'].mean()
            
            pb206_204SE_percent = data_ratios_['206Pb/204Pb'].sem()/pb206_204*100
            pb207_204SE_percent = data_ratios_['207Pb/204Pb'].sem()/pb207_204*100
            pb207_206SE_percent = data_ratios_['207Pb/206Pb'].sem()/pb207_206*100
            u238_235SE_percent = data_ratios_['238U/235U'].sem()/u238_235*100
            u238_th232SE_percent = data_ratios_['238U/232Th'].sem()/u238_th232*100
            
            
            ratios_to_return = [pb207_206,u238_235,u238_th232,pb206_204,pb207_204]
            stats_to_return = [pb207_206SE_percent,u238_235SE_percent,u238_th232SE_percent,pb206_204SE_percent,pb207_204SE_percent]
            
            return ratios_to_return,stats_to_return
        
    
    
    def get_regressions(data,regression_buttons,ablation_start_true):
        """
        function that gets the 206Pb/238U regression

        Parameters
        ----------
        data : dataframe
            pandas dataframe hosting the measured data.
        regression_buttons : string
            string object defining which type of regression to use.
        ablation_start_true : float
            float of where to project the ratio back to.

        Returns
        -------
        Returned objects depend on the function that calls get_regressions
            get_approved: returns regressed 206/238 and error
            ratio_plot: returns regression to plot
            get_regression stats: returns regression statistics to display them to user
            residuals_plot: returns regression residuals to plot

        """
        data = calc_fncs.threesig_outlierremoval(data)
        data = data.reset_index(drop=True)
        t0 = data.loc[0,'Time']
        data['Time'] = data['Time']-t0
        ablation_start_true = ablation_start_true-t0
        y = data['206Pb/238U']
        y207 = data['207Pb/235U']
        if '1st Order' in regression_buttons:
            y1, X1 = dmatrices('y ~ Time', data=data, return_type='dataframe') # get y and x regression data
            mod1 = sm.OLS(y1, X1) # fit a linear model on the data
            fit1 = mod1.fit() # get the list of fit parameters
            predicted1 = fit1.params[0] + fit1.params[1]*data.Time # get the predicted y values for the given x values
            rsquared1 = fit1.rsquared # get the R2 value of the regression
            predicted_b01 = fit1.params[0] + fit1.params[1]*ablation_start_true # get the predicted value at the ablation start that is input by the user
            sigma1 = np.sqrt(fit1.ssr/fit1.df_resid) # get the 1SD (Sum Squared residuals / residual degrees of freedom)^(1/2)
            # get standard error for a single point estiamted by a regression model
            SE_b01 = sigma1*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2)))
            SE_b01_percent = SE_b01/predicted_b01*100 # get the % 1SE
            resid1 = fit1.resid # get the residuals of the regression
            
            y1_207,X1_207 = dmatrices('y207 ~ Time', data=data, return_type='dataframe') # get y and x regression data
            fit1_207 = sm.OLS(y207,X1_207).fit()
            predicted1_207 = fit1_207.params[0] + fit1_207.params[1]*data.Time # get the predicted y values for the given x values
            predicted_b01_207 = fit1_207.params[0] + fit1_207.params[1]*ablation_start_true # get the predicted value at the ablation start that is input by the user
            sigma1_207 = np.sqrt(fit1_207.ssr/fit1_207.df_resid) # get the 1SD (Sum Squared residuals / residual degrees of freedom)^(1/2)
            SE_b01_207 = sigma1_207*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2)))
            SE_b01_percent_207 = SE_b01_207/predicted_b01_207*100 # get the % 1SE
            resid1_207 = fit1_207.resid # get the residuals of the regression
        else:
            # fill the above with blank values so that there is not an error in the output if '1st Order' is not wanted by the user
            predicted1 = np.zeros_like(data['Time'])
            rsquared1 = 0
            predicted_b01 = 0
            SE_b01 = 0
            SE_b01_percent = 0
            resid1 = np.zeros_like(data['Time'])
            
            predicted1_207 = np.zeros_like(data['Time'])
            predicted_b01_207 = 0
            SE_b01_207 = 0
            SE_b01_percent_207 = 0
            resid1_207 = np.zeros_like(data['Time'])
            
        if 'Exp. Regression' in regression_buttons:
            # fit exponential regression - ommitting added constant on the backend (Paton et al., 2010) as it is simply an unneeded added parameter that's difficult to interpret in log space
            def exp_func(x,a,b,c):
                return a*np.exp(-b*x)+c
            def simple_exp_func(x,a,b):
                return a*np.exp(-b*x)
            
            curve638 = 'extra variable'
            curve735 = 'extra variable'
            
            default_206238_initparams = [0.1,0.05,0.02]
            default_207235_initparams = [1,0.05,0.02]
            
            try:
                popt,pcov = curve_fit(exp_func,data['Time'].to_numpy(),data['206Pb/238U'].to_numpy(),p0=default_206238_initparams)
            except RuntimeError:
                print('Runtime Error: simplifying exponential function 206/238')
                newparams_638 = [0.1,0.05]
                popt,pcov = curve_fit(simple_exp_func,data['Time'].to_numpy(),data['206Pb/238U'].to_numpy(),p0=newparams_638)
                curve638 = 'simple'
                
            try:
                popt_207,pcov_207 = curve_fit(exp_func,data['Time'].to_numpy(),data['207Pb/235U'].to_numpy(),p0=default_207235_initparams)
            except RuntimeError:
                print('Runtime Error: simplifying exponential function 207/235')
                newparams_735 = [0.9,0.05]
                popt_207,pcov_207 = curve_fit(simple_exp_func,data['Time'].to_numpy(),data['207Pb/235U'].to_numpy(),p0=newparams_735)
                curve735 = 'simple'
                
            if curve638 == 'extra variable':
                predictedexp = exp_func(data['Time'],*popt)
                predicted_b0exp = popt[0]*np.exp(-popt[1]*ablation_start_true)+popt[2]
            elif curve638 == 'simple':
                predictedexp = simple_exp_func(data['Time'],*popt)
                predicted_b0exp = popt[0]*np.exp(-popt[1]*ablation_start_true)
            if curve735 == 'extra variable':
                predictedexp_207 = exp_func(data['Time'],*popt_207)
                predicted_b0exp_207 = popt_207[0]*np.exp(-popt_207[1]*ablation_start_true)+popt_207[2]
            elif curve735 == 'simple':
                predictedexp_207 = simple_exp_func(data['Time'],*popt_207)
                predicted_b0exp_207 = popt_207[0]*np.exp(-popt_207[1]*ablation_start_true)
                
            
            # initialize arrays to be filled with residuals and squared residuals
            resid = np.zeros(len(predictedexp))
            sq_resid = np.zeros(len(predictedexp))
            resid_207 = np.zeros(len(predictedexp))
            sq_resid_207 = np.zeros(len(predictedexp))
            # for i,k,m,l in zip(range(0,len(predictedexp)),range(0,len(data['206Pb/238U'])),range(0,len(predictedexp_207)),range(0,len(data['207Pb/235U']))):
            for i in range(0,len(predictedexp)):
                resid[i] = ((data['206Pb/238U'][i] - predictedexp[i])) # get residuals
                sq_resid[i] = (((data['206Pb/238U'][i] - predictedexp[i])**2)) # square them
                resid_207[i] = ((data['207Pb/235U'][i]-predictedexp_207[i]))
                sq_resid_207[i] = ((data['207Pb/235U'][i]-predictedexp_207[i])**2)
            sum_sq_resid = np.sum(sq_resid) # sum the squared residuals
            sum_sq_resid_207 = np.sum(sq_resid_207)
            sigmaexp = np.sqrt(sum_sq_resid/(len(data['206Pb/238U'])-2)) # denominator = d.f. = n-#params
            sigmaexp_207 = np.sqrt(sum_sq_resid_207/(len(data['207Pb/235U'])-2)) # denominator = d.f. = n-#params
            SE_b0exp = sigmaexp*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2))) # get standard error of intercept
            SE_b0exp_207 = sigmaexp_207*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2)))
            SE_b0exp_percent = SE_b0exp/predicted_b0exp*100 # calculate % error for SE of intercept
            SE_b0exp_percent_207 = SE_b0exp_207/predicted_b0exp_207*100
            residexp = resid # reassign variables
            residexp_207 = resid_207
            tss = ((data['206Pb/238U'] - np.mean(data['206Pb/238U']))**2).sum() # get the total sum of squared residuals
            rsquared_exp = 1 - (sum_sq_resid/tss) # get the r-squared for the exponential regression
            rsquaredexp_adj = 1 - ( (1-rsquared_exp)*(len(data['206Pb/238U'])-1) / (len(data['206Pb/238U'])-2-1) ) # get the adjusted r squared for the exponential regression
        else:
            # set everything to zeros if exponenetial regression not chosen
            predictedexp = np.zeros_like(data['Time'])
            rsquared_exp = 0
            rsquaredexp_adj = 0
            predicted_b0exp = 0
            SE_b0exp = 0
            SE_b0exp_percent = 0
            residexp = np.zeros_like(data['Time'])
            predictedexp_207 = np.zeros_like(data['Time'])
            predicted_b0exp_207 = 0
            SE_b0exp_207 = 0
            SE_b0exp_percent_207 = 0
            residexp_207 = np.zeros_like(data['Time'])
    
        # get the method that called up regressions. f_back gets the function that called. Removing this gives current method
        callingmethod = sys._getframe().f_back.f_code.co_name
        # set a series of if statements that causes the appropriate return depending on the function that called up regresssions
        if callingmethod == 'get_approved':
            
            # put the calculated values and statistics in lists to be returned
            ratios_to_return = [predicted_b01,predicted_b0exp,predicted_b01_207,predicted_b0exp_207]
            stats_to_return = [SE_b01_percent,SE_b0exp_percent,
                               SE_b01_percent_207,SE_b0exp_percent_207]
            
            return ratios_to_return,stats_to_return
        
        elif callingmethod == 'ratio_plot':
            regressions_to_return = [predicted1,predictedexp,predicted1_207,predictedexp_207]
            stats_to_report = [rsquared1,rsquaredexp_adj,SE_b01_percent,SE_b0exp_percent]
            
            return regressions_to_return,stats_to_report
        
        elif callingmethod == 'evaluate_output_data':
            # ""
            regressions_to_return = [predicted1,predictedexp]
            stats_to_report = [rsquared1,rsquared_exp,SE_b01_percent,SE_b0exp_percent,SE_b01_percent_207,SE_b0exp_percent_207]
            
            return regressions_to_return,stats_to_report
        
        elif callingmethod == 'residuals_plot':
            predicted_to_return = [predicted1,predictedexp]
            predicted_to_return_207 = [predicted1_207,predictedexp_207]
            resid_to_return = [resid1,residexp]
            resid_to_return_207 = [resid1_207,residexp_207]
            
            return predicted_to_return,predicted_to_return_207,resid_to_return,resid_to_return_207
        
        elif callingmethod == 'get_ellipse':
            predicted1_ellipse = resid1 + fit1.params[0]
            predicted1_207_ellipse = resid1_207 + fit1_207.params[0]
            if 'Exp. Regression' in regression_buttons:
                if curve638 == 'extra variable':
                    predictedexp_ellipse = residexp + popt[0] + popt[2]
                else:
                    predictedexp_ellipse = residexp + popt[0]
                if curve735 == 'extra variable':
                    predictedexp_207_ellipse = residexp_207 + popt_207[0] + popt_207[2]
                else:
                    predictedexp_207_ellipse = residexp_207 + popt_207[0]
            else:
                predictedexp_ellipse = np.zeros_like(data['Time'])
                predictedexp_207_ellipse = np.zeros_like(data['Time'])
            
            predicted_to_return = [predicted1_ellipse,predictedexp_ellipse]
            predicted_to_return_207 = [predicted1_207_ellipse,predictedexp_207_ellipse]
            
            return predicted_to_return,predicted_to_return_207

        
        else:
            pass
        
        
    def get_approved(data,bckgrnd_start_input,bckgrnd_stop_input,
                     ablation_start_input,ablation_stop_input,ablation_start_true,
                     regression_buttons,ellipsemode_selector,
                     counts_mode,integration_time,sample_name,new_index,power):
        """

        Parameters
        ----------
        data : dataframe
            pandas dataframe hosting the measured data.
        bckgrnd_start_input : integer
            integer from the slider defining when to start taking the gas blank.
        bckgrnd_stop_input : integer
            integer from the slider defining when to stop taking the gas blank.
        ablation_start_input : integer
            integer from the slider defining when to start the regression / counts.
        ablation_stop_input : integer
            integer form the slider defining when to stop the regression / counts.
        ablation_start_true : float
            float value of the projected ablation start time.
        regression_buttons : string
            string object defining which type of regression to use.
        ellipsemode_selector : Boolean
            Object defining weather or not to include confidence ellipses in output
        counts_mode : string
            string denoting which method the using wants to reduce data with.
        integration_time: float
            integration time used on the MS
        sample_name: string
            string input for the sample name input by user after hitting accept button
        new_index: integer
            integer value indexing the measurement number
            
        Returns
        -------
        data_approved : array
            arary of all reduced ratios and analytes.
        ellipse_data_toapprove : dataframe
            dataframe holding all the passes of the detector across the selected ablation period.

        """
        data_toapprove = data.reset_index(drop=True) # reset the index of the data so that it can be altered/indexed appropriately
        og_col_length = len(data_toapprove.columns[0:]) # get original column length
        analyte_cols = data_toapprove.columns[0:og_col_length-1] # get the column names of the analytes that will go into ratios
        # need to send background subtracted data to regression, so do the following:
        background = data_toapprove[(data_toapprove.Time >= bckgrnd_start_input) & (data_toapprove.Time <= bckgrnd_stop_input)] # get the measured background across the selected interval
        lod = [] # create a list for detection limits to be filled
        # create a for loop to calculate the detection limits: LOD = (3SD * 2^(1/2)) / n^(1/2) (Longerich et al., 1996)
        for i in background.columns[0:-1]:
            limit = 3*background[i].std()/np.sqrt(len(background[i]))*np.sqrt(2)
            lod.append(limit)
        background = background.mean()[0:-1] # calculate the mean background
        # subtract the mean background from all data. Any values < 0 are assigned a value of zero to avoid errors when passing dataframe through functions.
        # these are assigned as 'bdl' later
        background_subtracted_data = data_toapprove.iloc[:,0:-1].sub(background,axis='columns').clip(lower=0)
        # need to write a loop to deal with those passes on the detector where the value is below dl but not background before
        # calculating ratios
        
        
        data_toapprove.iloc[:,0:-1] = background_subtracted_data # insert the background subtracted data into the copied dataframe
        data_toapprove = data_toapprove[(data_toapprove.Time >= ablation_start_input) & (data_toapprove.Time <= ablation_stop_input)]
        data_toapprove = data_toapprove.reset_index(drop=True)
        
        if counts_mode == 'Total Counts':
            data_ratios_all,data_stats_all = calc_fncs.get_counts(data_toapprove, counts_mode, ablation_start_true, bckgrnd_start_input, bckgrnd_stop_input, ablation_start_input, ablation_stop_input,integration_time,analyte_cols)
            data_ratios = pd.DataFrame([data_ratios_all],columns=['206Pb/238U','238U/206Pb','207Pb/235U','208Pb/232Th','207Pb/206Pb','238U/235U','238U/232Th','206Pb/204Pb','207Pb/204Pb'])
            
            data_stats = pd.DataFrame([data_stats_all],columns=['SE% 206Pb/238U','SE% 238U/206Pb','SE% 207Pb/235U','SE% 208Pb/232Th',
                                                                'SE% 207Pb/206Pb','SE% 238U/235U','SE% 238U/232Th',
                                                                'SE% 206Pb/204Pb','SE% 207Pb/204Pb'])
            
        elif counts_mode == 'Means & Regression':
            data_approved_ratio = data_toapprove.copy() # copy the dataframe as to not overwrite the input data
            data_approved_ratio = calc_fncs.get_ratios(data_approved_ratio) # get calculated ratios of the background subtracted data. Need 0 and not 'bdl' for calcs to pass
        
            data_ratios_reg,data_stats_reg = calc_fncs.get_regressions(data_approved_ratio,regression_buttons,ablation_start_true) # get estimated regression intercepts, background subtracted ratios, regression statistics, and SE's of the ratios3
            data_ratios_counts,data_stats_counts = calc_fncs.get_counts(data_toapprove, counts_mode, ablation_start_true, bckgrnd_start_input, bckgrnd_stop_input, ablation_start_input, ablation_stop_input,integration_time,analyte_cols)
            data_ratios = data_ratios_reg + data_ratios_counts
            data_stats = data_stats_reg + data_stats_counts
            data_ratios = pd.DataFrame([data_ratios],columns=['206Pb/238U 1st Order','206Pb/238U Exp.','207Pb/235U 1st Order','207Pb/235U Exp.','207Pb/206Pb','238U/235U','238U/232Th','206Pb/204Pb','207Pb/204Pb']) # put the relevant calculations in df with appropriate column headers
            # put the relevant calculations in df with appropriate column headers
            data_stats = pd.DataFrame([data_stats],columns=['SE% 206Pb/238U 1st Order','SE% 206Pb/238U Exp','SE% 207Pb/235U 1st Order','SE% 207Pb/235U Exp',
                                                            'SE% 207Pb/206Pb','SE% 238U/235U','SE% 238U/232Th',
                                                            'SE% 206Pb/204Pb','SE% 207Pb/204Pb'])
        
        # get the ablation period indexed, background subtracted, indvidual isotope measurements. 
        # Copy data to ellipsoids before bdls are assigned if ellipsoids are indeed requested
        data_toapprove = data_toapprove[(data_toapprove.Time >= ablation_start_input) & (data_toapprove.Time <= ablation_stop_input)]
        if ellipsemode_selector is True:
            ellipse_data_toapprove = data_toapprove.copy()
            ellipse_data_toapprove = ellipse_data_toapprove.reset_index(drop=True)
            ellipse_data_toapprove = calc_fncs.get_ratios(ellipse_data_toapprove)
        data_toapprove_SE = pd.DataFrame([data_toapprove.iloc[:,0:-1].sem()]).add_suffix('_1SE') # get SE's of the individual isotopes
        data_toapprove = pd.DataFrame([data_toapprove.iloc[:,0:-1].mean()]) # put means of isotopic values in a dataframe
        
        # 202Hg: 29.86%, 204Hg: 6.87% https://www.nndc.bnl.gov/nudat3/
        Hgratio = (6.87/100) / (29.86/100) # get cononical 204Hg/202Hg ratio
        
        Weth_ellparams,TW_ellparams,x1,y1,y2 = calc_fncs.get_ellipse(ellipse_data_toapprove,power,ablation_start_true,regression_buttons,counts_mode)
        
        Weth_ellparams = pd.DataFrame([Weth_ellparams],columns=['Weth C','Weth Wid1','Weth Wid2','Weth rho'])
        TW_ellparams = pd.DataFrame([TW_ellparams],columns=['TW C','TW Wid1','TW Wid2','TW rho'])
        
        # needs to come after getting ellipsoid params because function uses get_ratios, which needs the data input in the form that it is uploaded
        
        # take the means of the individual isotopes and run them through a for loop with the detection limits. Assign a value of 'bdl' if below detection limit. Otherwise, leave the 
        # value unchanged
        for i,k in zip(range(0,len(data_toapprove.iloc[0])),lod):
            if data_toapprove.iloc[0,i]<k:
                data_toapprove.iloc[0,i]='bdl'
            elif data_toapprove.iloc[0,i]>k:
                data_toapprove.iloc[0,i]=data_toapprove.iloc[0,i]
                
        # subtract the isobaric interference of 204Hg on 204Pb using the measured 202Hg, so long as 202 > 'bdl', for the reduced data (i.e., the point estimtes - not every integration)
        if '202Hg' in data_toapprove.columns:
            if data_toapprove.loc[0,'202Hg'] != 'bdl':
                data_toapprove['204Hg'] = data_toapprove['202Hg']*Hgratio # calculate 204Hg from measured 202Hg based on isotopic abundance
                # subtract the 204Hg from the 204 signal, so long as the 204 signal > 'bdl'
                if data_toapprove.loc[0,'204Pb'] != 'bdl':
                    data_toapprove['204Pb'] = data_toapprove['204Pb'] - data_toapprove['204Hg']
                    # Recheck to make sure newly calculated if the newly calculated 204 signal is greater or less than 'bdl'. Assign 'bdl' or leave unchanged appropriately.
                    loc204 = data_toapprove.columns.get_loc('204Pb')
                    if data_toapprove.iloc[0,loc204] <= lod[loc204]:
                        data_toapprove['204Pb'] = 'bdl'
                    elif data_toapprove.iloc[0,loc204] > lod[loc204]:
                        data_toapprove['204Pb'] = data_toapprove['204Pb']
            else:
                data_toapprove['204Hg'] = 'bdl'
        
        data_toapprove.insert(0,'measurementindex',[new_index])
        data_toapprove.insert(1,'SampleLabel',[sample_name]) # reinsert the sample label into the calculation df
        data_toapprove.insert(2,'t start',[ablation_start_input]) # insert ablation start time into df
        data_toapprove.insert(3,'t end',[ablation_stop_input]) # insert ablation stop time into df
        data_toapprove.insert(4,'t project',[ablation_start_true]) # insert projected regression start time into df
        data_toapprove.insert(5,'b start',[bckgrnd_start_input])
        data_toapprove.insert(6,'b end',[bckgrnd_stop_input])
        # stitch the individual isotopic measurements, their errors, ratios, ratio errors, and regression results / regression statistics into a df together.
        # these are then appeneded into the output df
        data_approved = data_toapprove.join([data_toapprove_SE,data_ratios,data_stats,Weth_ellparams,TW_ellparams])
        
        # check if any of the individual analytes in their reduced form (i.e., their means) are bdl. if so, set the ratio to bdl as well
        ratio_cols = ['206Pb/204Pb','207Pb/204Pb','207Pb/206Pb','238U/235U']
        for i in analyte_cols:
            for k in ratio_cols:
                if k.__contains__(str(i)) and data_approved[i].item() == 'bdl':
                    data_approved[k] = 'bdl'
                else:
                    pass
        
        
        return data_approved
    
    
    def get_ellipse(data,power,ablation_start_true,regression_buttons,counts_mode):
        """
        Function that gets the confidence ellipses

        Parameters
        ----------
        data : dataframe
            pandas dataframe holding the observed data.
        power : float
            float object defining the power of the confidence. Recommended to keep this at 0.05

        Returns
        -------
        ell1_params : array
            array of float objects defining the dimensions of the confidence ellipse for Wetherhill concordia.
        ell2_params : array
            array of float objects defining the dimensions of the confidence ellipse for Tera-Wasserburg concordia.

        """
        data = data.dropna()
        drop_condn = data[(data['206Pb/238U'] == 0) | (data['207Pb/235U'] == 0) | (data['207Pb/206Pb'] == 0)].index
        data.drop(drop_condn,inplace=True)
        data = data.reset_index(drop=True)
        
        if counts_mode != 'Total Counts':
            adjusted,adjusted_207 = calc_fncs.get_regressions(data,regression_buttons,ablation_start_true)
            if ('1st Order' in regression_buttons) and ('Exp. Regression' not in regression_buttons):
                data['207Pb/235U'] = adjusted_207[0]
                data['206Pb/238U'] = adjusted[0]
            else:
                data['207Pb/235U'] = adjusted_207[1]
                data['206Pb/238U'] = adjusted[1]
        
        x1 = data['207Pb/235U']
        y1 = data['206Pb/238U']
        x2 = 1/data['206Pb/238U']
        y2 = data['207Pb/206Pb']
        
        cov1 = np.cov(x1,y1)
        cov2 = np.cov(x2,y2)
        eigval1,eigvec1 = np.linalg.eig(cov1)
        eigval2,eigvec2 = np.linalg.eig(cov2)
        order1 = eigval1.argsort()[::-1]
        order2 = eigval2.argsort()[::-1]
        eigvals_order1 = eigval1[order1]
        eigvals_order2 = eigval2[order2]
        eigvecs_order1 = eigvec1[:,order1]
        eigvecs_order2 = eigvec2[:,order2]
        
        c1 = (np.mean(x1),np.mean(y1))
        c2 = (np.mean(x2),np.mean(y2))
        wid1 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power),df=2)*eigvals_order1[0])
        hgt1 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power),df=2)*eigvals_order1[1])
        wid2 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power),df=2)*eigvals_order2[0])
        hgt2 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power),df=2)*eigvals_order2[1])
        theta1 = np.degrees(np.arctan2(*eigvecs_order1[:,0][::-1]))
        theta2 = np.degrees(np.arctan2(*eigvecs_order2[:,0][::-1]))
        
        ell1_params = [c1,wid1,hgt1,theta1]
        ell2_params = [c2,wid2,hgt2,theta2]
        
        return ell1_params,ell2_params,x1,y1,y2
    
        
        
class plots(calc_fncs):
    """ Class that holds all of the functions for reducing the time resolved data"""
    def __init__(self,*args):
        super().__init__(*args)
        for a in args:
            self.__setattr__(str(a), args[0])
            
    
    def ratio_plot(data,ratio_buttons,regression_buttons,
                   start_bckgrnd,stop_bckgrnd,
                   start_ablation,stop_ablation,ablation_start_true,displayframe):
        """
        Function for displaying the plot of relavent isotopic ratios.
        
        Parameters
        ----------
        data : pandas dataframe
            pandas dataframe of observed data
        ratio_buttons : list of strings
            list of strings that hosts the ratios the user requests to plot
        regression_buttons : list of strings
            list of strings that hoststhe regressions the user requests
        start_bckgrnd : integer
            Integer value input by the user in the background_slider param. This is the lower value on the slider 
        stop_bckgrnd : integer
            Integer value input by the user in the background_slider param. This is the higher value on the slider
        start_ablation : integer
            Integer value input by the user in the ablation_slider param. This is the lower value on the slider
        stop_ablation : integer
            Integer value input by the user in the ablation_slider param. This is the higher value on the slider
        ablation_start_true : integer
            integer value input by the user in the ablation_start_true param. Regression is projected back to this value
    
        Returns
        -------
        fig : bokeh figure
            plot that shows the relevant isotopic ratios for the entire measurement (background+ablation+washout)
    
        """
        data = calc_fncs.get_ratios(data) # get calculated ratios from the data
        data_to_regress = data[(data.Time >= start_ablation) & (data.Time <= stop_ablation)] # get the selected ablation period
        if displayframe == 'main':
            height,width = 350,600
        else:
            height,width = 500,500
        fig = figure(height=height,width=width,title='Isotope Ratios',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='above',
                     x_axis_label='Time (s)',
                     y_range=[min(data_to_regress['206Pb/238U'])-min(data_to_regress['206Pb/238U'])*0.1,max(data_to_regress['206Pb/238U'])+max(data_to_regress['206Pb/238U'])*0.1])
        var_cols = ratio_buttons # get the ratios selected by the user. These are used to get the columns from the calculated ratios
        regressions_to_plot = regression_buttons# get the regression type requested by the user
        # get regression parameters and stats for the requested regressions across the specified ablation period
        # plot a line for each selected ratio
        for i,c in zip(var_cols,cycle(color_palette)):
            fig.line(data.Time,data[i],line_width=0.7,legend_label='{}'.format(i),color=c)
        # plot a line for each selected regression
        if displayframe == 'main':
            pass
        else:
            regressions,stats = calc_fncs.get_regressions(data_to_regress,regression_buttons,ablation_start_true)
            if regressions_to_plot is not None:
                for i,c in zip(regressions,cycle(color_palette_regressions)):
                    fig.line(data_to_regress.Time,i,line_width=0.5,color=c)
        fig.line([start_bckgrnd,start_bckgrnd],[0,1],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background start
        fig.line([stop_bckgrnd,stop_bckgrnd],[0,1],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background stop
        fig.line([start_ablation,start_ablation],[0,1],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation start
        fig.line([stop_ablation,stop_ablation],[0,1],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation stop
        fig.line([ablation_start_true,ablation_start_true],[0,1],line_width=0.4,line_dash='dotted',color='black') # vertical dotted line for ablation start true
        
        return fig
    
    
    def ratio_plot_76(data,
                      start_bckgrnd,stop_bckgrnd,
                      start_ablation,stop_ablation,ablation_start_true):
        """
        Function for displaying the plot of relavent isotopic ratios.
        
        Parameters
        ----------
        data : pandas dataframe
            pandas dataframe of observed data
        ratio_buttons : list of strings
            list of strings that hosts the ratios the user requests to plot
        regression_buttons : list of strings
            list of strings that hoststhe regressions the user requests
        start_bckgrnd : integer
            Integer value input by the user in the background_slider param. This is the lower value on the slider 
        stop_bckgrnd : integer
            Integer value input by the user in the background_slider param. This is the higher value on the slider
        start_ablation : integer
            Integer value input by the user in the ablation_slider param. This is the lower value on the slider
        stop_ablation : integer
            Integer value input by the user in the ablation_slider param. This is the higher value on the slider
        ablation_start_true : integer
            integer value input by the user in the ablation_start_true param. Regression is projected back to this value
    
        Returns
        -------
        fig : bokeh figure
            plot that shows the relevant isotopic ratios for the entire measurement (background+ablation+washout)
    
        """
        data = calc_fncs.get_ratios(data) # get calculated ratios from the data
        data_to_regress = data[(data.Time >= start_ablation) & (data.Time <= stop_ablation)] # get the selected ablation period
        fig = figure(height=350,width=600,title='Isotope Ratios',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='above',
                     x_axis_label='Time (s)',
                     y_range=[min(data_to_regress['207Pb/206Pb'])-min(data_to_regress['207Pb/206Pb'])*0.1,max(data_to_regress['207Pb/206Pb'])+max(data_to_regress['207Pb/206Pb'])*0.1])
        fig.line(data.Time,data['207Pb/206Pb'],line_width=0.7,color='teal')
        fig.line([start_bckgrnd,start_bckgrnd],[0,1],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background start
        fig.line([stop_bckgrnd,stop_bckgrnd],[0,1],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background stop
        fig.line([start_ablation,start_ablation],[0,1],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation start
        fig.line([stop_ablation,stop_ablation],[0,1],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation stop
        fig.line([ablation_start_true,ablation_start_true],[0,1],line_width=0.4,line_dash='dotted',color='black') # vertical dotted line for ablation start true
        
        return fig
    
    def ablation_plot(data_ablation_,bckgrnd_start_input,bckgrund_stop_input,ablation_start_input,ablation_stop_input,
                      ablation_start_true,logdata,analytes_):  
        """
        Function to plot the measured isotopic data across the entire analysis (background+ablation+washout)
    
        Parameters
        ----------
        data_ablation_ : pandas dataframe
            dataframe of the measured values.
        bckgrnd_start_input : integer
            lower integer value of the background_slider param.
        bckgrund_stop_input : integer
            upper integer value of the background_slider param.
        ablation_start_input : integer
            lower value of the abaltion_slider param.
        ablation_stop_input : integer
            upper value of the ablation_slider param.
        ablation_start_true : integer
            integer value that indicates where the regression will be projected back towards (i.e., start of ablation).
        logdata: boolean
            allows user to choose if data should be on log scale
        analytes: object
            list of individual analytes to include on ablation plot
    
        Returns
        -------
        fig : bokeh fig
            Figure showing all of the time resolved ablation data.
    
        """
        # assign variable y_type to log data or not, based on boolean input
        if logdata == True:
            y_type = 'log'
        else:
            y_type = 'auto'
        # intialize figure
        data_in_frame = data_ablation_[(data_ablation_['Time']>=bckgrnd_start_input) & (data_ablation_['Time']<=ablation_stop_input)]
        max_list = []
        min_list = []
        for a in analytes_:
            max_list.append(max(data_in_frame[a]))
            min_list.append(min(data_in_frame[a]))
        
        min_val = min(min_list)
        max_val = max(max_list)
        fig = figure(height=350,width=1200,title='All Time Resolved Data',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='left',
                      y_axis_type=y_type,x_axis_label = 'Time (s)', y_axis_label = 'Intensities (cps)',
                      x_range=[bckgrnd_start_input-3,ablation_stop_input+10],y_range=[min_val-min_val*0.05,max_val+max_val*0.01]
                      )
        var_cols=analytes_ # reassign anlytes_ variable
        # zip colors and analytes together to get plotted vs time
        for i,c in zip(var_cols,cycle(color_palette)):
            fig.line(data_ablation_.Time,data_ablation_[i],line_width=0.7,legend_label='{}'.format(i),color=c)
        
        fig.line([bckgrnd_start_input,bckgrnd_start_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background start
        fig.line([bckgrund_stop_input,bckgrund_stop_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='dashed',color='black') # vertical dashed line for background stop
        fig.line([ablation_start_input,ablation_start_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation start
        fig.line([ablation_stop_input,ablation_stop_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation stop
        fig.line([ablation_start_true,ablation_start_true],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='dotted',color='black') # vertical dotted line for projection location
        
        return fig
        
        
    def residuals_plot(data,regression_buttons,start_ablation,stop_ablation,ablation_start_true):
        """
        Function to plot residuals from regressed time resolved ratios

        Parameters
        ----------
        data : pandas dataframe
            dataframe containing data between the ablation interval
        regression_buttons : list of strings
            list of regressions to be visualized
        start_ablation : float
            value determining where the ablation starts
        stop_ablation : float
            value determining where the ablation ends
        ablation_start_true : float
            value determining where the regression is projected back to

        Returns
        -------
        fig : bokeh figure
            bokeh figure containing the plotted residuals. Two plots: upper is 206/238 residuals for all regressions requested, lower is same for 207/235

        """
        fig206 = figure(height=250,width=250,title='206Pb/238U Residuals',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='above',
                        x_axis_label='Fitted Value',y_axis_label='Residuals')
        fig207 = figure(height=250,width=250,title='207Pb/235U Residuals',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='above',
                        x_axis_label='Fitted Value',y_axis_label='Residuals')
        data = calc_fncs.get_ratios(data) # calculate relevant isotopic ratios from the data
        data_to_regress = data # reassign variable name for clarity
        # get fitted regression values, regression names, and residuals
        fitted,fitted207,residuals,residuals207 = calc_fncs.get_regressions(data_to_regress,regression_buttons,ablation_start_true)
        # plot the residuals for the regressions and the fitted value
        for j,c in zip(range(0,len(regression_buttons)),color_palette_regressions):
            fig206.circle(fitted[j],residuals[j],color=c,legend_label='{}'.format(regression_buttons[j]))
            fig207.circle(fitted207[j],residuals207[j],color=c,legend_label='{}'.format(regression_buttons[j]))
        fig206.line([min(fitted[j]),max(fitted[j])],[0,0],color='black',line_width=0.4)
        fig207.line([min(fitted207[j]),max(fitted207[j])],[0,0],color='black',line_width=0.4)
        fig = gridplot([[fig206],[fig207]],width=500,height=250)
        
        return fig

    
    def ellipse_plot(data,power,start_ablation,stop_ablation,ablation_start_true,regression_buttons,counts_mode):
        """
        Function that creates ellipsoids from data in the ablation interval and plots them

        Parameters
        ----------
        data : pandas dataframe
            dataframe containing the data between the selected ablation interval
        power : float
            'power' of the confidence ellipse
        start_ablation : float
            starting time for the ablation
        stop_ablation : float
            ending time for the ablation

        Returns
        -------
        fig1 : matplotlib figure
            figure containing the ellipse and integrations corresponding to Wetherhill concordia.
        fig2 : matplotlib figure
            figure containing the ellipse and integrations corresponding to Tera-Wasserburg concordia

        """
        data = calc_fncs.get_ratios(data)
        data = data.dropna()
        drop_condn = data[(data['206Pb/238U'] == 0) | (data['207Pb/235U'] == 0) | (data['207Pb/206Pb'] == 0)].index # set up a mask to drop observations that are = 0
        data.drop(drop_condn,inplace=True) # drop them
        ell1p,ell2p,pbu735,pbu638,pbpb76 = calc_fncs.get_ellipse(data, power,ablation_start_true,regression_buttons,counts_mode) # get the parameters of the confidence ellipsoids

        ell1 = Ellipse(xy=ell1p[0],width=ell1p[1],height=ell1p[2],angle=ell1p[3],color='darkkhaki',ec='k',alpha=0.5) # set the parameters into a plotable 'patch'
        ell2 = Ellipse(xy=ell2p[0],width=ell2p[1],height=ell2p[2],angle=ell2p[3],color='darkkhaki',ec='k',alpha=0.5)
        fig1 = Figure(figsize=(3,3)) # create a figure that is 8Wx8H
        fig2 = Figure(figsize=(3,3))
        ax1 = fig1.add_subplot() # add an axis to the mpl figure
        ax2 = fig2.add_subplot()
        
        ax1.add_artist(ell1) # adde the ellipsoid patch to the axis
        ax1.plot(pbu735,pbu638,'.k',markersize=1) # plot individual observations as dots
        ax2.add_artist(ell2)
        ax2.plot(1/pbu638,pbpb76,'.k',markersize=1)
        
        ax1.set_xlabel('207/235',fontsize=6) # set xlabel
        ax1.set_ylabel('206/238',fontsize=6) # set ylabel
        ax2.set_xlabel('238/206',fontsize=6)
        ax2.set_ylabel('207/206',fontsize=6)
        ax1.tick_params(axis='both',labelsize=5) # set tick parameters on the axes
        ax2.tick_params(axis='both',labelsize=5)
        ax1.set_xlim(ell1p[0][0]-ell1p[1]/1.5,ell1p[0][0]+ell1p[1]/1.5) # set reasonable x and y limits based on the size of hte patch
        ax1.set_ylim(ell1p[0][1]-ell1p[2]/1.5,ell1p[0][1]+ell1p[2]/1.5)
        ax2.set_xlim(ell2p[0][0]-ell2p[1]/1.5,ell2p[0][0]+ell2p[1]/1.5)
        ax2.set_ylim(ell2p[0][1]-ell2p[2]/1.5,ell2p[0][1]+ell2p[2]/1.5)
        
        fig1.tight_layout() 
        fig2.tight_layout()
        
        return fig1,fig2

    

# %%
class make_plots(param.Parameterized):
    """ class that parameterizes inputs and sends them to the above functions to be rendered in a GUI"""
    update_output_button = param.Action(lambda x: x.evaluate_output_data(),label='Evaluate Interval') # Button that triggers function to add output data
    export_data_button = param.Action(lambda x: x.export_data(),label='DDDT!') # button that triggers function to export the reduced data
    jump_forward_button = param.Action(lambda x: x.jump_sliders(),label='Jump Sliders') # button that triggers sliders to jump forward

    lock_ablation_start_true = param.Boolean(True,label='Lock Back Projection')
    ablation_start_true = param.Number(30.9) # parameterize a number that defines where regressions get projected back to for t0 intercept
    jump_time = param.Number(53) # number that defines how far to jump sliders when function is triggered
    
    # set up list of clickable buttons to choose which ratios gets plotted
    ratio_buttons = param.ListSelector(default=['206Pb/238U'], objects=['206Pb/238U','207Pb/235U','208Pb/232Th','207Pb/206Pb','238U/235U','206Pb/204Pb'])
    # set up list of clickable buttons to choose which regressions to use
    regression_buttons = param.ListSelector(default=['1st Order'], objects=['1st Order','Exp. Regression'])
    
    logcountsdata = param.Boolean(False,label='Log Intensities') # set up boolean button to choose whether or not 
    analytes_ = param.ListSelector(default=[],objects=[]) # set up empty list to be populated with analytes in order to choose which gets plotted
    # list of buttons to choose whether to reduce using total counts or means + regression
    counts_mode = param.Selector(default='Total Counts',objects=['Total Counts','Means & Regression']) 
    # integration time variable. Has a default but gets reassigned based on Nu output file
    integration_time = param.Number(default=0.01)
    # boolean button to choose whether or not to calculate confidence ellipsoids
    ellipsemode_selector = param.Boolean(True,label='Generate Ellipse')
    power = param.Number(default=0.05) #power for confidence ellipse
    
    input_data = param.DataFrame(precedence=-1) # initialize dataframe to be populated with uploaded data
    file_path = param.String(default='Insert File Path') # string that will be populated with file path
    output_data = param.DataFrame(precedence=-1) # initialize dataframe to be populated with output data
    output_data_ellipse = param.DataFrame(precedence=-1) # initialize dataframe to be populated with ellipsoid output data
    
    accept_array_button = param.Action(lambda x: x.close_modal_setdata(),label='Accept Detector Array') # button that triggers the collector block anaalyte assignments to be accepted
    accept_interval_button = param.Action(lambda x: x.send_reduction(),label='Accept Interval') # button that triggers sample name to be accepted and ablation to be reduced
    ablation_start = param.Number(47.1,bounds=(0,8600),softbounds=(50,90),step=0.1) # number that defines where ablation intervals starts
    ablation_end = param.Number(73.3,bounds=(0,8600),step=0.1) # number that defines where ablation ends
    background_start = param.Number(26.2,bounds=(0,8600),step=0.1) # number that defines where background starts
    background_end = param.Number(43.3,bounds=(0,8600),step=0.1) # number that defines where background ends
    
        
    def __init__(self,**params):
        super().__init__(**params)
        self.file_input_widget = pn.Param(self.param.input_data)
        self.output_data_widget = pn.Param(self.param.output_data)
        self.output_data_ellipse_widget = pn.Param(self.param.output_data_ellipse)
        self.widgets = pn.Param(self,parameters=['update_output_button','export_data_button','lock_ablation_start_true',
                                                 'ablation_start_true','jump_time',
                                                 'logcountsdata','analytes_',
                                                 'ratio_buttons','regression_buttons','counts_mode',
                                                 'ellipsemode_selector','power',
                                                 'integration_time','file_path',
                                                 'ablation_start','ablation_end','background_start','background_end',
                                                 ])
    
    @pn.depends('file_path',watch=True)
    def _uploadfile(self):
        """
        Function that handles the output Nu file and uploads raw data into software
        -------
        Opens a modal that allows the user to input their collector array

        """
        if self.file_path != 'Insert File Path':
            df = pd.read_excel(self.file_path,header=None) # read file
            self.integration_time = df.iloc[69,1] # grab the integration time used for the analytical session
            self.input_data = df # assign input file to the initialized dataframe
            self.input_data.columns = self.input_data.iloc[76] # assign columns to the list of high/lowmass column (e.g., L204) output by the Nu
            self.input_data = self.input_data.drop(columns=['Cycle','Section','Type','Trigger Status']) # drop columns with these titles
            self.input_data = self.input_data.drop(self.input_data.index[0:77],axis=0) # drop rows with metadata
            self.input_data = self.input_data.reset_index(drop=True) # reset indices after dropping
            # self.input_data.drop(self.input_data.index[0:75],axis=0)
            n_analytes = len(self.input_data.columns)-1 # get the number of analytes from the number of active collectors
            # set up variables to be updated in a loop. These append the same number of strings and buttons to the modal as there are active collectors
            nth_iter = 0
            rth_row = 0
            for n in range(n_analytes):
                # loop through rows and columns, putting in a string and button to select Faraday or IC (i.e., whether data is in volts or counts, repsectively)
                fastgrid_layout.modal[rth_row].append(pn.Column(pn.Row(pn.widgets.TextInput(placeholder='Mass')),
                                                                pn.Row(pn.widgets.RadioButtonGroup(options=['Faraday','IC']))))
                nth_iter = nth_iter + 1
                if nth_iter % 3 == 0:
                    rth_row = rth_row + 1
            fastgrid_layout.modal[-1].append(pn.Column(buttons_))
            fastgrid_layout.open_modal()
        
    @pn.depends('ablation_start_true','background_start','background_end','ablation_start','ablation_end','jump_time','lock_ablation_start_true')
    def jump_sliders(self,event=None, watch=True):
        """
        Function that advances ('jumps') sliders forward based on the jump time

        """
        if self.lock_ablation_start_true == True:
            self.ablation_start_true = self.ablation_start+self.jump_time
        else:
            self.ablation_start_true = self.ablation_start_true+self.jump_time
        self.ablation_end = self.ablation_end+self.jump_time
        self.ablation_start = self.ablation_start+self.jump_time
        self.background_end = self.background_end+self.jump_time
        self.background_start = self.background_start+self.jump_time
        
        
    @pn.depends('input_data','background_start','background_end','ablation_start','ablation_end','ablation_start_true','analytes_','lock_ablation_start_true')
    def call_ablation_plot(self):
        """
        Function that calls and places the plot with time resolved analytes into a bokeh pane

        Returns
        -------
        bokeh pane
            hosts figure

        """
        if self.lock_ablation_start_true == True:
            self.ablation_start_true = self.ablation_start
        if self.output_data is not None:
            data_toplot = self.input_data
            return pn.pane.Bokeh(row(plots.ablation_plot(data_toplot,self.background_start,self.background_end,
                                                         self.ablation_start,self.ablation_end,self.ablation_start_true,
                                                         self.logcountsdata,self.analytes_)))
    
        
    @pn.depends('input_data','ratio_buttons','regression_buttons','background_start','background_end','ablation_start','ablation_end','ablation_start_true','lock_ablation_start_true')
    def call_ratio_plot(self):
        """
        Function that calls and places the plot with time resolved ratios into a bokeh pane

        Returns
        -------
        bokeh pane
            hosts figure

        """
        if self.output_data is not None:
            if self.lock_ablation_start_true == True:
                self.ablation_start_true = self.ablation_start
            data_toplot = self.input_data[(self.input_data['Time'] >= self.background_start-3) & (self.input_data['Time'] <= self.ablation_end+10)]
            return pn.pane.Bokeh(row(plots.ratio_plot(data_toplot,self.ratio_buttons,self.regression_buttons,self.background_start,self.background_end,
                                                      self.ablation_start,self.ablation_end,self.ablation_start_true,'main')))
        
    @pn.depends('input_data','background_start','background_end','ablation_start','ablation_end','ablation_start_true','lock_ablation_start_true')
    def call_ratio_plot76(self):
        """
        Function that calls and places the plot with time resolved ratios into a bokeh pane

        Returns
        -------
        bokeh pane
            hosts figure

        """
            
        if self.output_data is not None:
            if self.lock_ablation_start_true == True:
                self.ablation_start_true = self.ablation_start
            data_toplot = self.input_data[(self.input_data['Time'] >= self.background_start-3) & (self.input_data['Time'] <= self.ablation_end+10)]
            return pn.pane.Bokeh(row(plots.ratio_plot_76(data_toplot,
                                                         self.background_start,self.background_end,
                                                         self.ablation_start,self.ablation_end,self.ablation_start_true)))

        

    @pn.depends('output_data',watch=True)
    def _update_output_widget(self):
        """
        Function that displays output data when updated

        Returns
        -------
        Tabulator table
            hosts output data

        """
        if self.output_data is not None:
            self.output_data_widget = self.output_data
            self.output_data_widget.height = 400
            self.output_data_widget.heightpolicy = 'Fixed'
            return pn.widgets.Tabulator(self.output_data_widget,width=600) # use 600 for large screen, 100-150 for small screen 
    
    
    @pn.depends('input_data','ratio_buttons','regression_buttons',
                'background_start','background_end','ablation_start','ablation_end','ablation_start_true','lock_ablation_start_true',
                'power','counts_mode')
    def evaluate_output_data(self, event=None):
        """
        Function that clears any residuals input from uploading data or previously approved analyses, then generates fresh ones in a modal

        Parameters
        ----------
        event : open panel modal

        """
        
        if self.ablation_start_true == True:
            self.ablation_start_true = self.ablation_start
            
        data_ratio_plot = self.input_data[(self.input_data['Time'] >= self.background_start-5) & (self.input_data['Time'] <= self.ablation_end+10)]
        ratioplot_pane = pn.pane.Bokeh(row(plots.ratio_plot(data_ratio_plot,self.ratio_buttons,self.regression_buttons,self.background_start,self.background_end,
                                                  self.ablation_start,self.ablation_end,self.ablation_start_true,'modal')))
        
        data_residuals_plot = self.input_data[(self.input_data['Time'] >= self.ablation_start) & (self.input_data['Time'] <= self.ablation_end)]
        residuals_plot_pane = pn.pane.Bokeh(row(plots.residuals_plot(data_residuals_plot,self.regression_buttons,self.ablation_start,self.ablation_end,self.ablation_start_true)))
        
        Weth_ell,TW_ell = plots.ellipse_plot(data_residuals_plot, self.power, self.ablation_start, self.ablation_end, self.ablation_start_true, self.regression_buttons,self.counts_mode)
            
        ellipse_tabs = pn.Tabs(('TW',TW_ell),('Weth.',Weth_ell),dynamic=True)
        
        
        datastats = calc_fncs.get_ratios(data_residuals_plot)
        datastats = datastats.reset_index(drop=True)
        regressions,stats = calc_fncs.get_regressions(datastats,self.regression_buttons,self.ablation_start_true)
        divider1 = pn.pane.Markdown('206/238 Reg. Stats:')
        r2_1 = pn.pane.Markdown('$$R^{2}$$ = '+str(round(stats[0],3)))
        r2_exp = pn.pane.Markdown('$$R^{2}_{exp}$$ = '+str(round(stats[1],3)))
        SE_1stper = pn.pane.Markdown('$$1^{st} Order SE %% $$ = '+str(round(stats[2],3)))
        SE_expper = pn.pane.Markdown('$$Exp SE %% $$ = '+str(round(stats[3],3)))
        divider2 = pn.pane.Markdown('207/235 Reg. Stats:')
        SE_1stper_207 = pn.pane.Markdown('$$1^{st} Order SE %% $$ = '+str(round(stats[4],3)))
        SE_expper_207 = pn.pane.Markdown('$$Exp SE %% $$ = '+str(round(stats[5],3)))
         
        stats_markdown = pn.Row(pn.Column(divider1,r2_1,r2_exp,SE_1stper,SE_expper),pn.Column(divider2,SE_1stper_207,SE_expper_207))
        
        
        fastgrid_layout.modal[0].clear()
        fastgrid_layout.modal[1].clear()
        fastgrid_layout.modal[2].clear()
        fastgrid_layout.modal[3].clear()
        fastgrid_layout.modal[4].clear()
        
        fastgrid_layout.modal[0].append(buttons_sample)
        fastgrid_layout.modal[1].append(pn.widgets.TextInput(placeholder='Enter Sample Name'))
        fastgrid_layout.modal[2].append(pn.Row(ratioplot_pane,pn.Spacer(width=50, height=500),residuals_plot_pane))
        fastgrid_layout.modal[3].append(pn.Row(ellipse_tabs,pn.Spacer(width=50, height=500),stats_markdown))
        
        fastgrid_layout.open_modal()
    
        
    def send_reduction(self,event=None):
        """
        Function that gets the fully reduced data and sends it to the output data file that will be exported. Closes modal.

        """
        new_index = 0
        sample_name = fastgrid_layout.modal[1][0].value
        data_tosend = self.input_data[(self.input_data['Time'] >= self.background_start) & (self.input_data['Time'] <= self.ablation_end)] # filter data for background through ablation interval
        if self.lock_ablation_start_true == True:
            self.ablation_start_true = self.ablation_start
        # get the approved data by calling functions
        data_approved = calc_fncs.get_approved(data_tosend,self.background_start,self.background_end,self.ablation_start,self.ablation_end,
                                                                   self.ablation_start_true,self.regression_buttons,self.ellipsemode_selector,
                                                                   self.counts_mode,self.integration_time,sample_name,new_index,self.power)
        # if there is no data in the first column (assuming you can see 238U) assign the first sample measurement as -1 and make this global measurement (measurmentindex) 1
        # otherwise, add 1 to the last global measurement number and to the last sample measurement number
        if self.output_data.loc[0,'238U'] <= 0.0:
            updated_sample_name = sample_name+str('-1')
            data_approved['SampleLabel'] = updated_sample_name
            self.output_data = data_approved
        else:
            update_index = self.output_data['measurementindex'].iloc[-1] + 1
            data_approved['measurementindex'] = update_index
            if self.output_data['SampleLabel'].str.contains(sample_name).any():
                duplicates = self.output_data[self.output_data['SampleLabel'].str.contains(sample_name)]
                added_amount = int(duplicates['SampleLabel'].iloc[-1].rsplit('-')[-1]) + 1
                updated_sample_name = sample_name+str('-')+str(added_amount)
            else:
                updated_sample_name = sample_name+str('-1')
            data_approved['SampleLabel'] = updated_sample_name
            self.output_data = pd.concat([self.output_data,data_approved],ignore_index=True)
    
        fastgrid_layout.close_modal()
 
        
    @pn.depends('analytes_')
    def close_modal_setdata(self,event=None):
        """
        Function triggered that accepts input collector block and inputs necessary columns for recording metadata during reduction. Closes modal

        """
        n_analytes = len(self.input_data.columns)-1
        nth_iter = 0
        nth_iter_col = 0
        rth_row = 0
        added_analytes_list = []
        for n in range(n_analytes):
            next_analyte = fastgrid_layout.modal[rth_row][nth_iter][0][0].value
            detector_type = fastgrid_layout.modal[rth_row][nth_iter][1][0].value
            self.input_data = self.input_data.rename(columns={self.input_data.columns[nth_iter_col+1]: next_analyte})
            added_analytes_list.append(next_analyte)
            if detector_type == 'Faraday':
                self.input_data[next_analyte] = self.input_data[next_analyte] / volt_count_constant
            elif detector_type == 'IC':
                pass
            nth_iter = nth_iter + 1
            nth_iter_col = nth_iter_col + 1
            if nth_iter % 3 == 0:
                rth_row = rth_row + 1
                nth_iter = 0    
        self.param.analytes_.objects = added_analytes_list            
        self.input_data['Time'] = self.input_data['Time Stamp (S)']
        self.input_data = self.input_data.drop('Time Stamp (S)',axis=1)
        self.input_data['Time'] = self.input_data['Time'].astype('float')
        self.input_data = self.input_data.reset_index(drop=True)
        self.output_data = pd.DataFrame([np.zeros(len(self.input_data.columns))],columns=list(self.input_data.columns))
        self.output_data.insert(0,'measurementindex',0)
        self.output_data.insert(1,'SampleLabel',0)
        self.output_data.insert(2,'t start',0)
        self.output_data.insert(3,'t end',0)
        self.output_data.insert(4,'t project',0)
        self.output_data.insert(5,'b start',0)
        self.output_data.insert(6,'b end',0)
        # analytelength = len(self.input_data.columns)-1
        if self.ellipsemode_selector is True:
            self.output_data_ellipse = pd.DataFrame([np.zeros(len(self.input_data.columns))],columns=list(self.input_data.columns))
            self.output_data_ellipse.insert(0,'measurementindex',0)
            self.output_data_ellipse.insert(1,'SampleLabel',0)
            self.output_data_ellipse.insert(2,'t start',0)
            self.output_data_ellipse.insert(3,'t end',0)
            self.output_data_ellipse.insert(4,'t project',0)
            self.output_data_ellipse.insert(5,'b start',0)
            self.output_data_ellipse.insert(4,'b end',0)
        else:
            pass
        fastgrid_layout.close_modal()
        
        
    @pn.depends('output_data','output_data_ellipse')
    def export_data(self,event=None):
        """
        Function that exports the reduced data in an excel file

        Parameters
        ----------
        event : pd.to_excel

        """
        self.output_data.to_excel('output_lasertramZ.xlsx')
        if self.ellipsemode_selector is True and self.output_data_ellipse is not None:
            self.output_data_ellipse.to_excel('output_CEllipse_lasertramZ.xlsx')
        else:
            pass


# %%
callapp = make_plots(name='Reduce Ablation Data')

pn.extension('tabulator','mathjax')

buttons_=pn.WidgetBox(pn.Param(callapp.param.accept_array_button,
                               widgets={'accept_array_button': pn.widgets.Button(name='Accept Detector Array',button_type='success')}))
buttons_sample=pn.WidgetBox(pn.Param(callapp.param.accept_interval_button,
                               widgets={'accept_interval_button': pn.widgets.Button(name='Accept Sample Name',button_type='success')}))


widgets={'ratio_buttons': pn.widgets.CheckBoxGroup,
         'regression_buttons': pn.widgets.CheckBoxGroup,
         'counts_mode': pn.widgets.RadioButtonGroup,
         'export_data_button': pn.widgets.Button(name='DDDT!',button_type='success'),
         'analytes_': pn.widgets.CheckBoxGroup,
         'ablation_start': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(47.1),step=0.1,name='Ablation Start'),
         'ablation_end': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(73.3),step=0.1,name='Ablation End'),
         'background_start': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(26.2),step=0.1,name='Background Start'),
         'background_end': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(43.3),step=0.1,name='Background End')
         }

    
fastgrid_layout = pn.template.VanillaTemplate(title='LaserTRAMZ: LA-MC-ICP-MS',
                                                sidebar=pn.Column(pn.WidgetBox(pn.Param(callapp.param,widgets=widgets))),sidebar_width=380)

fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())

fastgrid_layout.main.append(pn.Row(pn.WidgetBox(pn.Param(callapp.param.ablation_start,widgets={'ablation_start': pn.widgets.EditableFloatSlider(start=0,end=7600,value=(51.1),step=1,name='Ablation Start',width=1300,height=20)},),
                                                pn.Param(callapp.param.ablation_end,widgets={'ablation_end': pn.widgets.EditableFloatSlider(start=0,end=7600,value=(76.3),step=1,name='Ablation End',width=1300,height=20),}),
                                                pn.Param(callapp.param.background_start,widgets={'background_start': pn.widgets.EditableFloatSlider(start=0,end=7600,value=(31.2),step=1,name='Background Start',width=1300,height=20),}),
                                                pn.Param(callapp.param.background_end,widgets={'background_end': pn.widgets.EditableFloatSlider(start=0,end=7600,value=(46.3),step=1,name='Background End',width=1300,height=20)}),
                                                width=1400,height=180
                                                )
                                   )
                            ) # for vanilla
fastgrid_layout.main.append(pn.Column(callapp.call_ablation_plot,pn.Row(callapp.call_ratio_plot,callapp.call_ratio_plot76))) # for vanilla
fastgrid_layout.main.append(pn.Column(callapp._update_output_widget))

fastgrid_layout.show();


    

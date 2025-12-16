#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 15:06:27 2025

@author: ctlewis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from scipy.optimize import curve_fit
import sys
from bokeh.plotting import figure, show

volt_count_constant = 1.602e-8 #volts / count

data = pd.read_excel('/Users/ctlewis/Documents/Projects/Chaxas/zircon_data/NuZirconData_032024/MCzrndata_03202024/Runone_03202024/runone_all_03202024.xlsx',header=None)
integration_time = data.iloc[69,1] # grab the integration time used for the analytical session
data.columns = data.iloc[76] # assign columns to the list of high/lowmass column (e.g., L204) output by the Nu
data = data.drop(columns=['Cycle','Section','Type','Trigger Status']) # drop columns with these titles
data = data.drop(data.index[0:77],axis=0) # drop rows with metadata
data = data.reset_index(drop=True) # reset indices after dropping
cols = ['Time','238U','235U','232Th','208Pb','207Pb','206Pb','204Pb','202Hg']
data.columns = cols

data.iloc[:,1:] = data.iloc[:,1:]/volt_count_constant
integration_time = 0.1

start,end = 150,225
datatoplot = data[(data['Time']>=start) & (data['Time']<=end)].reset_index(drop=True)

plt.plot(datatoplot['Time'],datatoplot['238U'],'-r')

bstart,bend = 159,171
tstart,tend = 178,202

for a in data.loc[:,'238U':'202Hg'].columns:
    print(a)

for analyte in data.columns[1:]:
    print(analyte)


def backgroundsubtract_convert(data,bstart,bend,tstart,tend,integrationtime):
    """
    Function for getting background subtracted counts data. 
    Takes raw intensities in cps, background subtracts the intensities, then converts to counts based on the integration time
    
    Parameters
    ----------
    data : pandas dataframe
        dataframe of time resolved intensities
    bstart : float
        user selected value for the background interval start
    bend : float
        user selected value for the background interval end
    tstart : float
        user selected value for the ablation interval start
    tend : float
        user selected value for the ablation interval end
    integrationtime : float
        integration time - stripped from the input file

    Returns
    -------
    ablation_backsub_counts : pandas dataframe
        pandas dataframe of background subtracted data converted into counts

    """
    backgrounds = data[(data['Time']>=bstart) & (data['Time']<=bend)]
    ablation = data[(data['Time']>=tstart) & (data['Time']<=tend)]
    


    meanbackgrounds = backgrounds.mean()[1:]

    ablation_backsub = ablation.iloc[:,1:].sub(meanbackgrounds,axis='columns').clip(lower=0)
    ablation_backsub_counts = ablation_backsub*integration_time
    ablation_backsub_counts.insert(0,'Time',ablation['Time'])
    ablation_backsub_counts = ablation_backsub_counts.reset_index(drop=True)
    
    return ablation_backsub_counts,meanbackgrounds

newdata,meanbackgrounds = backgroundsubtract_convert(data, bstart, bend, tstart, tend, integration_time)




def get_mean_ratios(data,ratiotype):
    """
    Function used to get reduced isotope ratios of data. Values are done with either ratio of means method or geometric mean
    Uncertainties are calculated according to selection  - standard error of time resolved ratio or geometric standard error of time resolved ratio

    Parameters
    ----------
    data : pandas dataframe
        dataframe containing values to reduce. Must be counts data as output from the function backgroundsubtract_convert()
    ratiotype : string
        string value specifying which method will be used to reduce ratios.
        Must be either 'Ratio of Means' or 'Geometric'

    Returns
    -------
    reduced_ratios : pandas dataframe
        dataframe with reduced ratios and uncertainties in 1SE%

    """
    if ratiotype == 'Ratio of Means':
        mu_206Pb238U = np.mean(data['206Pb'])/np.mean(data['238U']) if np.mean(data['238U'])>0 else 0
        se_206Pb238U = 0 if mu_206Pb238U==0 else np.std(np.divide(data['206Pb'].astype(float),data['238U'].astype(float),out=np.zeros_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        mu_238U206Pb = np.mean(data['238U'])/np.mean(data['206Pb']) if np.mean(data['206Pb'])>0 else 0
        se_238U206Pb = 0 if mu_238U206Pb==0 else np.std(np.divide(data['238U'].astype(float),data['206Pb'].astype(float),out=np.zeros_like(data['238U'].astype(float)),where=data['206Pb'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        try:
            mu_207Pb235U = np.mean(data['207Pb'])/np.mean(data['235U']) if np.mean(data['235U'])>0 else np.mean(data['207Pb'])/np.mean(data['238U']/137.818)
            se_207Pb235U = np.std(np.divide(data['207Pb'].astype(float),data['238U'].to_numpy()/137.818,out=np.zeros_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0),ddof=1)/np.sqrt(len(data)) if np.mean(data['235U'])<=0 else np.std(np.divide(data['207Pb'].astype(float),data['235U'].astype(float),out=np.zeros_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        except:
            mu_207Pb235U = 0
            se_207Pb235U = 0
        mu_208Pb232Th = np.mean(data['208Pb'])/np.mean(data['232Th']) if np.mean(data['232Th'])>0 else 0
        se_208Pb232Th = 0 if mu_208Pb232Th==0 else np.std(np.divide(data['208Pb'].astype(float),data['232Th'].astype(float),out=np.zeros_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        mu_207Pb206Pb = np.mean(data['207Pb'])/np.mean(data['206Pb']) if np.mean(data['206Pb'])>0 else 0
        se_207Pb206Pb = 0 if mu_207Pb206Pb==0 else np.std(np.divide(data['207Pb'].astype(float),data['206Pb'].astype(float),out=np.zeros_like(data['206Pb'].astype(float)),where=data['206Pb'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        mu_207Pb204Pb = np.mean(data['207Pb'])/np.mean(data['204Pb']) if np.mean(data['204Pb'])>0 else 0
        se_207Pb204Pb = 0 if mu_207Pb204Pb==0 else np.std(np.divide(data['207Pb'].astype(float),data['204Pb'].astype(float),out=np.zeros_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0))/np.sqrt(len(data))
        mu_206Pb204Pb = np.mean(data['206Pb'])/np.mean(data['204Pb']) if np.mean(data['204Pb'])>0 else 0
        se_206Pb204Pb = 0 if mu_206Pb204Pb==0 else np.std(np.divide(data['206Pb'].astype(float),data['204Pb'].astype(float),out=np.zeros_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        mu_238U232Th = np.mean(data['238U'])/np.mean(data['232Th']) if np.mean(data['232Th'])>0 else 0
        se_238U232Th = 0 if mu_238U232Th==0 else np.std(np.divide(data['238U'].astype(float),data['232Th'].astype(float),out=np.zeros_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        mu_238U235U = np.mean(data['238U'])/np.mean(data['235U']) if np.mean(data['235U'])>0 else 137.818
        se_238U235U = 137.818 if mu_238U235U==0 else np.std(np.divide(data['238U'].astype(float),data['235U'].astype(float),out=np.zeros_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0),ddof=1)/np.sqrt(len(data))
        
    elif ratiotype == 'Geometric':
        mu_206Pb238U = st.gmean(np.divide(data['206Pb'].astype(float),data['238U'].astype(float),out=np.ones_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0)) if np.mean(data['238U'])>0 else 0
        se_206Pb238U = 0 if mu_206Pb238U==0 else st.gstd(np.divide(data['206Pb'].astype(float),data['238U'].astype(float),out=np.ones_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0))/np.sqrt(len(data))
        mu_238U206Pb = 1/mu_206Pb238U
        se_238U206Pb = se_206Pb238U
        try:
            mu_207Pb235U = st.gmean(np.divide(data['207Pb'].astype(float),data['235U'].astype(float),out=np.ones_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0)) if np.mean(data['235U'].astype(float))>0 else st.gmean(np.divide(data['207Pb'].astype(float),data['238U'].to_numpy()/137.818,out=np.ones_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0))
            se_207Pb235U = st.gstd(np.divide(data['207Pb'].astype(float),data['238U'].to_numpy()/137.818,out=np.ones_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0)) if np.mean(data['235U'].astype(float))<=0 else st.gstd(np.divide(data['207Pb'].astype(float),data['235U'].astype(float),out=np.ones_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0))/np.sqrt(len(data))
        except:
            mu_207Pb235U = 0
            se_207Pb235U = 0
        mu_208Pb232Th = st.gmean(np.divide(data['208Pb'].astype(float),data['232Th'].astype(float),out=np.ones_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0)) if np.mean(data['232Th'])>0 else 0
        se_208Pb232Th = 0 if mu_208Pb232Th==0 else st.gstd(np.divide(data['208Pb'].astype(float),data['232Th'].astype(float),out=np.ones_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0))/np.sqrt(len(data))
        mu_207Pb206Pb = st.gmean(np.divide(data['207Pb'].astype(float),data['206Pb'].astype(float),out=np.ones_like(data['206Pb'].astype(float)),where=data['206Pb'].astype(float)!=0)) if np.mean(data['206Pb'])>0 else 0
        se_207Pb206Pb = 0 if mu_207Pb206Pb==0 else st.gstd(np.divide(data['207Pb'].astype(float),data['206Pb'].astype(float),out=np.ones_like(data['206Pb'].astype(float)),where=data['206Pb'].astype(float)!=0))/np.sqrt(len(data))
        mu_207Pb204Pb = st.gmean(np.divide(data['207Pb'].astype(float),data['204Pb'].astype(float),out=np.ones_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0)) if np.mean(data['204Pb'])>0 else 0
        se_207Pb204Pb = 0 if mu_207Pb204Pb==0 else st.gstd(np.divide(data['207Pb'].astype(float),data['204Pb'].astype(float),out=np.ones_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0))/np.sqrt(len(data))
        mu_206Pb204Pb = st.gmean(np.divide(data['206Pb'].astype(float),data['204Pb'].astype(float),out=np.ones_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0)) if np.mean(data['204Pb'])>0 else 0
        se_206Pb204Pb = 0 if mu_206Pb204Pb==0 else st.gstd(np.divide(data['206Pb'].astype(float),data['204Pb'].astype(float),out=np.ones_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0))/np.sqrt(len(data))
        mu_238U232Th = st.gmean(np.divide(data['238U'].astype(float),data['232Th'].astype(float),out=np.ones_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0)) if np.mean(data['232Th'])>0 else 0
        se_238U232Th = 0 if mu_238U232Th==0 else st.gstd(np.divide(data['238U'].astype(float),data['232Th'].astype(float),out=np.ones_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0))/np.sqrt(len(data))
        mu_238U235U = st.gmean(np.divide(data['238U'].astype(float),data['235U'].astype(float),out=np.ones_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0)) if np.mean(data['235U'])>0 else 137.818
        se_238U235U = 137.818 if mu_238U235U==0 else st.gstd(np.divide(data['238U'].astype(float),data['235U'].astype(float),out=np.ones_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0))/np.sqrt(len(data))
        
    means_array = np.array([mu_206Pb238U,mu_238U206Pb,mu_207Pb235U,mu_208Pb232Th,mu_207Pb206Pb,mu_207Pb204Pb,mu_206Pb204Pb,mu_238U232Th,mu_238U235U])
    uncertainties_array = np.array([se_206Pb238U,se_238U206Pb,se_207Pb235U,se_208Pb232Th,se_207Pb206Pb,se_207Pb204Pb,se_206Pb204Pb,se_238U232Th,se_238U235U])
    uncertainties_array = uncertainties_array/means_array*100
    full_array = np.concatenate((means_array,uncertainties_array))
    ratio_uncertainties_list = ['206Pb/238U','238U/206Pb','207Pb/235U','208Pb/232Th','207Pb/206Pb','207Pb/204Pb','206Pb/204Pb','238U/232Th','238U/235U',
                                '206Pb/238U SE%','238U/206Pb SE%','207Pb/235U SE%','208Pb/232Th SE%','207Pb/206Pb SE%','207Pb/204Pb SE%','206Pb/204Pb SE%','238U/232Th SE%','238U/235U SE%'
                                ]
    reduced_ratios = pd.DataFrame([full_array],columns=ratio_uncertainties_list)
    
    return reduced_ratios
        

# ratiotype = 'Ratio of Means'
# nratios = get_mean_ratios(newdata,ratiotype)


# ratiotype = 'Geometric'
# gratios = get_mean_ratios(newdata, ratiotype)


# fig,ax = plt.subplots(1,1,figsize=(8,5))
# ax.plot(newdata['Time'],newdata['207Pb']/newdata['206Pb'],'-k',lw=0.5)

# np.std(newdata['207Pb']/newdata['206Pb'])*3+np.mean(newdata['207Pb']/newdata['206Pb'])



def threesigoutlierremoval(data):
    
    # implement calling method to handle only relevant ratios is doing regression - all ratios if anything else
    ratios = ['206Pb/238U','238U/206Pb','207Pb/235U','208Pb/232Th','207Pb/206Pb','207Pb/204Pb','206Pb/204Pb','238U/232Th','238U/235U']
    tresolvedratios = get_tresolved_ratios(data)
    
    for r in ratios:
        threesig = 3*np.std(tresolvedratios[r])
        mean = np.mean(tresolvedratios[r])
        trigger = True
        while trigger is True:
            mask = (tresolvedratios[r] < mean - threesig) | (tresolvedratios[r] > mean + threesig)
            tresolvedratios.loc[mask,r] = np.nan
            tresolvedratios[r] = tresolvedratios[r].infer_objects(copy=False).interpolate(method='linear')
            threesig = 3*np.std(tresolvedratios[r])
            mean = np.mean(tresolvedratios[r])
            if any(mask):
                pass
            else:
                break
                
    return tresolvedratios


newdf = threesigoutlierremoval(newdata)



def get_tresolved_ratios(data):
    """
    function that returns time resolved ratios for the entire analysis from background start to ablation end
    used for visualization and thre sig outlier removal
    
    Parameters
    ----------
    data : pandas dataframe
        dataframe of time resolved intensities
    bstart : float
        user selected value for the background interval start
    tend : float
        user selected value for the ablation interval end

    Returns
    -------
    tresolvedr : pandas dataframe
        dataframe with time resovled ratios

    """
    r206238 = np.divide(data['206Pb'].astype(float),data['238U'].astype(float),out=np.zeros_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0) if np.mean(data['238U']>0) else np.zeros_like(data.iloc[:,1])
    r238206 = np.divide(data['238U'].astype(float),data['206Pb'].astype(float),out=np.zeros_like(data['206Pb'].astype(float)),where=data['206Pb'].astype(float)!=0) if np.mean(data['206Pb']>0) else np.zeros_like(data.iloc[:,1])
    r207235 = np.divide(data['207Pb'].astype(float),data['235U'].astype(float),out=np.zeros_like(data['235U'].astype(float)),where=data['235U'].astype(float)!=0) if np.mean(data['235U']>0) else np.divide(data['207Pb'].astype(float),(data['238U']/137.818).astype(float),out=np.zeros_like(data['238U'].astype(float)),where=data['238U'].astype(float)!=0) if np.mean(data['238U']>0) else np.zeros_like(data.iloc[:,1])
    r208232 = np.divide(data['208Pb'].astype(float),data['232Th'].astype(float),out=np.zeros_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0) if np.mean(data['232Th']>0) else np.zeros_like(data.iloc[:,1])
    r207206 = np.divide(data['207Pb'].astype(float),data['206Pb'].astype(float),out=np.zeros_like(data['206Pb'].astype(float)),where=data['206Pb'].astype(float)!=0) if np.mean(data['206Pb']>0) else np.zeros_like(data.iloc[:,1])
    r207204 = np.divide(data['207Pb'].astype(float),data['204Pb'].astype(float),out=np.zeros_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0) if np.mean(data['204Pb']>0) else np.zeros_like(data.iloc[:,1])
    r206204 = np.divide(data['206Pb'].astype(float),data['204Pb'].astype(float),out=np.zeros_like(data['204Pb'].astype(float)),where=data['204Pb'].astype(float)!=0) if np.mean(data['204Pb']>0) else np.zeros_like(data.iloc[:,1])
    r238232 = np.divide(data['238U'].astype(float),data['232Th'].astype(float),out=np.zeros_like(data['232Th'].astype(float)),where=data['232Th'].astype(float)!=0) if np.mean(data['232Th']>0) else np.zeros_like(data.iloc[:,1])
    r238235 = np.divide(data['238U'].astype(float),data['235U'].astype(float),out=np.zeros_like(data['235U'].astype(float)),where=data['238U'].astype(float)!=0) if np.mean(data['235U']>0) else np.full_like(data.iloc[:,1], 137.818)
 
    tresolved_ratio_list = [r206238,r238206,r207235,r208232,r207206,r207204,r206204,r238232,r238235]
    ratiolist = ['206Pb/238U','238U/206Pb','207Pb/235U','208Pb/232Th','207Pb/206Pb','207Pb/204Pb','206Pb/204Pb','238U/232Th','238U/235U']
    stacked_tresolvedr = np.stack(tresolved_ratio_list,axis=-1)
    tresolvedr = pd.DataFrame(stacked_tresolvedr,columns=ratiolist)
    tresolvedata = pd.concat([data.reset_index(drop=True),tresolvedr.reset_index(drop=True)],axis=1).reset_index(drop=True)
    
    return tresolvedata
 

# tresolved_df = get_tresolved_ratios(data)

# Note: Input data must be counts data only - no ratios output from the function used to get ratios
def get_regressions(data,regression_buttons,ablation_start_true):
    """
    Function used to get regressions of time-resolved Pb/U data. 
    Returns either 1st order regression or exponential regresssion
    Stats and regression parameters returned depends on calling function

    Parameters
    ----------
    data : pandas dataframe
        pandas dataframe of coutns data. Note that this cannot be data that already has ratios in the dataframe as these are calculated here
    regression_buttons : list
        list of arguments specifying if the user wants 1st order or exp. regression. Currently get 1st or both - need to implement way to get only one
    ablation_start_true : float
        float value indicating where to set time zero intercept in ablation. by default set to tstart in program

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    data = data.reset_index(drop=True)
    data = threesigoutlierremoval(data)
    t0 = data.loc[0,'Time']
    data['Time'] = data['Time']-t0
    ablation_start_true = ablation_start_true-t0
    y = data['206Pb/238U'].to_numpy(dtype=float)
    y207 = data['207Pb/235U'].to_numpy(dtype=float)
    x = data['Time'].to_numpy(dtype=float)
    X = sm.add_constant(x)
    if '1st Order' in regression_buttons:
        # 206Pb/238U 1st order regression
        linmod1 = sm.OLS(y, X).fit() # fit a linear model on the data
        predicted1 = linmod1.predict() # predicted values from y=mx+b
        rsquared1 = linmod1.rsquared # get the R2 value of the regression
        predicted_b01 = linmod1.params[0] + linmod1.params[1]*ablation_start_true # get the predicted value at the ablation start that is input by the user - ntoe this allows projection if desired - otherwise could use fit.params[0] or fit.params['const']
        sigma1 = np.sqrt(linmod1.ssr/linmod1.df_resid) # get the 1SD (Sum Squared residuals / residual degrees of freedom)^(1/2)
        SE_b01 = sigma1*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2))) # get standard error for a single point estiamted by a regression model
        SE_b01_percent = SE_b01/predicted_b01*100 # get the % 1SE
        resid1 = linmod1.resid # get the residuals of the regression
        # 207Pb/235U 1st order regression - equations and functions as above
        linmod1_207 = sm.OLS(y207,X).fit()
        predicted1_207 = linmod1_207.predict()
        predicted_b01_207 = linmod1_207.params[0] + linmod1_207.params[1]*ablation_start_true
        sigma1_207 = np.sqrt(linmod1_207.ssr/linmod1_207.df_resid) 
        SE_b01_207 = sigma1_207*np.sqrt(1/len(data)+(data['Time'].mean())**2/((len(data)-1)*np.var(data['Time'],ddof=2)))
        SE_b01_percent_207 = SE_b01_207/predicted_b01_207*100 
        resid1_207 = linmod1_207.resid
        
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
        # define the exponential function adn a simplified exponential function if iterations fail
        def exp_func(x,a,b,c):
            return a*np.exp(-b*x)+c
        def simple_exp_func(x,a,b):
            return a*np.exp(-b*x)
        # define a string variable to recognize if runtime error was excepted or not
        curve638 = 'Three Variables'
        curve735 = 'Three Variables'
        
        default_206238_initparams = [0.1,0.05,0.02]
        default_207235_initparams = [1,0.05,0.02]
        
        try:
            popt,pcov = curve_fit(exp_func,data['Time'].to_numpy(dtype=float),data['206Pb/238U'].to_numpy(dtype=float),p0=default_206238_initparams) # fit data to an exponential curve
        except RuntimeError:
            print('Runtime Error: simplifying exponential function 206/238')
            newparams_638 = [0.1,0.05]
            popt,pcov = curve_fit(simple_exp_func,data['Time'].to_numpy(dtype=float),data['206Pb/238U'].to_numpy(dtype=float),p0=newparams_638) # fit data to two parameter exponential curve in case runtime error occurred
            curve638 = 'Two Variables'
        except Exception as e:
            print('206Pb/238U Exp. regression failing due to: '+print(str(e)))
            
        try:
            popt_207,pcov_207 = curve_fit(exp_func,data['Time'].to_numpy(dtype=float),data['207Pb/235U'].to_numpy(dtype=float),p0=default_207235_initparams)
        except RuntimeError:
            print('Runtime Error: simplifying exponential function 207/235')
            newparams_735 = [0.9,0.05]
            popt_207,pcov_207 = curve_fit(simple_exp_func,data['Time'].to_numpy(dtype=float),data['207Pb/235U'].to_numpy(dtype=float),p0=newparams_735)
            curve735 = 'Two Variables'
        except Exception as e:
            print('207Pb/235U Exp. regression failing due to: '+print(str(e)))
            
        if curve638 == 'Three variables':
            predictedexp = exp_func(data['Time'].to_numpy(dtype=float),*popt) # get predicted values for exponential curve
            predicted_b0exp = popt[0]*np.exp(-popt[1]*ablation_start_true)+popt[2] # get zero-intercept for exponential curve
        elif curve638 == 'Two Variables':
            predictedexp = simple_exp_func(data['Time'].to_numpy(dtype=float),*popt)
            predicted_b0exp = popt[0]*np.exp(-popt[1]*ablation_start_true)
        if curve735 == 'Three Variables':
            predictedexp_207 = exp_func(data['Time'].to_numpy(dtype=float),*popt_207)
            predicted_b0exp_207 = popt_207[0]*np.exp(-popt_207[1]*ablation_start_true)+popt_207[2]
        elif curve735 == 'Two Variables':
            predictedexp_207 = simple_exp_func(data['Time'].to_numpy(dtype=float),*popt_207)
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
        
    return_predicted_stuff_to_test = [predicted_b01,predicted_b0exp,predicted_b01_207,predicted_b0exp_207,
                                      predicted1,predictedexp,predicted1_207,predictedexp_207
                                      ]
    return_uncertainty_stuff_to_test = [SE_b01_percent,SE_b0exp_percent,SE_b01_percent_207,SE_b0exp_percent_207,
                                        rsquared1,rsquaredexp_adj,resid1,residexp,resid1_207,residexp_207
                                        ]
    exp_params_to_test = [popt,pcov,popt_207,pcov_207,sum_sq_resid,sum_sq_resid_207]
    lin_params_to_test = [linmod1,linmod1_207]
    
    return return_predicted_stuff_to_test,return_uncertainty_stuff_to_test,exp_params_to_test,lin_params_to_test

    # # get the method that called up regressions. f_back gets the function that called. Removing this gives current method
    # callingmethod = sys._getframe().f_back.f_code.co_name
    # # set a series of if statements that causes the appropriate return depending on the function that called up regresssions
    # if callingmethod == 'get_approved':
        
    #     # put the calculated values and statistics in lists to be returned
    #     ratios_to_return = [predicted_b01,predicted_b0exp,predicted_b01_207,predicted_b0exp_207]
    #     stats_to_return = [SE_b01_percent,SE_b0exp_percent,
    #                        SE_b01_percent_207,SE_b0exp_percent_207]
        
    #     return ratios_to_return,stats_to_return
    
    # elif callingmethod == 'ratio_plot':
    #     regressions_to_return = [predicted1,predictedexp,predicted1_207,predictedexp_207]
    #     stats_to_report = [rsquared1,rsquaredexp_adj,SE_b01_percent,SE_b0exp_percent]
        
    #     return regressions_to_return,stats_to_report
    
    # elif callingmethod == 'evaluate_output_data':
    #     # ""
    #     regressions_to_return = [predicted1,predictedexp]
    #     stats_to_report = [rsquared1,rsquared_exp,SE_b01_percent,SE_b0exp_percent,SE_b01_percent_207,SE_b0exp_percent_207]
        
    #     return regressions_to_return,stats_to_report
    
    # elif callingmethod == 'residuals_plot':
    #     predicted_to_return = [predicted1,predictedexp]
    #     predicted_to_return_207 = [predicted1_207,predictedexp_207]
    #     resid_to_return = [resid1,residexp]
    #     resid_to_return_207 = [resid1_207,residexp_207]
        
    #     return predicted_to_return,predicted_to_return_207,resid_to_return,resid_to_return_207
    
    # elif callingmethod == 'get_ellipse':
    #     predicted_to_return = [predicted1,predictedexp]
    #     predicted_to_return_207 = [predicted1_207,predictedexp_207]
        
    #     return predicted_to_return,predicted_to_return_207
    
    # else:
    #     pass
    


newdata = backgroundsubtract_convert(data, bstart, bend, tstart, tend, integration_time)
tresolved_countsdata = get_tresolved_ratios(newdata)
tresolved_countsdata_outlierremoved = threesigoutlierremoval(newdata)
tresolved_ablation = get_tresolved_ratios(datatoplot)

regression_buttons = ['1st Order','Exp. Regression']
ablation_start_true = tstart
regression_predictedstuff, regression_uncertaintiesstuff,expparamstuff,linparamsstuff = get_regressions(newdata, regression_buttons, ablation_start_true)


    # return_predicted_stuff_to_test = [predicted_b01,predicted_b0exp,predicted_b01_207,predicted_b0exp_207,
    #                                   predicted1,predictedexp,predicted1_207,predictedexp_207
    #                                   ]
    # return_uncertainty_stuff_to_test = [SE_b01_percent,SE_b0exp_percent,SE_b01_percent_207,SE_b0exp_percent_207,
    #                                     rsquared1,rsquaredexp_adj,resid1,residexp,resid1_207,residexp_207
    #                                     ]
    # exp_params_to_test = [popt,pcov,popt_207,pcov_207]
    # lin_params_to_test = [linmod1,linmod1_207]


firstorder_LCI = linparamsstuff[0].get_prediction().summary_frame(alpha=0.05)["mean_ci_upper"]
firstorder_UCI = linparamsstuff[0].get_prediction().summary_frame(alpha=0.05)["mean_ci_lower"]


tresolved_countsdata_outlierremoved['Time'] = tresolved_countsdata_outlierremoved['Time']-min(tresolved_countsdata_outlierremoved['Time'])
tresolved_countsdata_outlierremoved['Time'].to_numpy(dtype=float)
tresolved_countsdata_outlierremoved['206Pb/238U'].to_numpy(dtype=float)

exp_UCI = []
exp_LCI = []
aBlist = []
x2 = tresolved_countsdata_outlierremoved['Time'].to_numpy(dtype=float).reshape(-1,1)
x1 = np.ones_like(x2)
X = np.hstack((x1,x2))
k = len(expparamstuff[0])
n = len(x1)
talpha2 = st.t.ppf(1-0.025,n-k)
SSE = expparamstuff[4]
S = np.sqrt(SSE/(n-k))


for x in range(0,len(tresolved_countsdata_outlierremoved)):
    a = np.array([1,tresolved_countsdata_outlierremoved.loc[x,'Time']])
    sqrtterm = a.T @ np.linalg.inv(np.matmul(X.T,X)) @ a
    CI = talpha2*S*np.sqrt(sqrtterm)
    U_CI_xi = regression_predictedstuff[5][x] + CI
    L_CI_xi = regression_predictedstuff[5][x] - CI
    exp_UCI.append(U_CI_xi)
    exp_LCI.append(L_CI_xi)
    


popt = expparamstuff[0]
pcov = expparamstuff[1]

new_exp_UCI = []
new_exp_LCI = []
for x in range(0,len(tresolved_countsdata_outlierremoved)):
    xi = tresolved_countsdata_outlierremoved.loc[x,'Time']
    Gprime1 = np.exp(-popt[1]*xi) # dx/da of function = G(•)' of delta method
    Gprime2 = popt[0]*xi*np.exp(-popt[1]*xi) # dx/db for delta method
    Gprime = np.array([Gprime1,Gprime2]) # make vector of G prime
    varB = expparamstuff[1] # covariance matrix of the parameters
    # varG = Gprime @ np.linalg.inv(np.matmul(varB.T,varB)) @ Gprime.T # for lin regression need X'X - not in delta method
    varG = Gprime @ varB @ Gprime.T
    sdG = np.sqrt(varG)
    CI = sdG * S * talpha2
    UCI_i = regression_predictedstuff[5][x] + CI
    LCI_i = regression_predictedstuff[5][x] - CI
    new_exp_UCI.append(UCI_i)
    new_exp_LCI.append(LCI_i)



fig = figure(height=500,width=800,title='Pb/U Regression Fits',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='above',
             x_axis_label='Time (s)',)
             # y_range=[min(tresolved_countsdata_outlierremoved['206Pb/238U'])-min(tresolved_countsdata_outlierremoved['206Pb/238U'])*0.1,
             #          max(tresolved_countsdata_outlierremoved['206Pb/238U'])+max(tresolved_countsdata_outlierremoved['206Pb/238U'])*0.1])
fig.line(tresolved_countsdata_outlierremoved['Time'],tresolved_countsdata_outlierremoved['206Pb/238U'],color='black')
fig.line(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[4],color='blue')
# fig.varea(tresolved_countsdata_outlierremoved['Time'],firstorder_LCI,firstorder_UCI,color='blue',alpha=0.3)
fig.line(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[5],color='peru')
fig.varea(tresolved_countsdata_outlierremoved['Time'],exp_LCI,exp_UCI,color='peru',alpha=0.3)
fig.varea(tresolved_countsdata_outlierremoved['Time'],new_exp_LCI,new_exp_UCI,color='magenta',alpha=0.3)

show(fig)




# use the puromycin example from BAtes and Watts to demonstrate correct parameterization
puromycinx = np.array([.02,.02,.06,.06,.11,.11,.22,.22,.56,.56,1.1,1.1]) # xdata
puromyciny = np.array([76,47,97,107,123,139,159,152,191,201,207,200]) # ydata

# define non-linear function describing reaction data
def MichaelisMenton(x,t1,t2):
    func = t1*x/(t2+x)
    return func
# define partial derivatives of the function with respect to the two parameters
def dmm_t1(x,t2):
    return x/(t2+x)
def dmm_t2(x,t1,t2):
    return -t1*x/((t2+x))**2

# fit the curve to get estimated parameter vector, var-covar matrix, R, other stats
curve_dict = curve_fit(MichaelisMenton,puromycinx,puromyciny,full_output=True) 
dir(curve_dict)
curve_dict

thetahat = curve_dict[0]  # get estimated parameters (theta hat)
varcov_mm = curve_dict[1] # get variance-covariance matrix
t1,t2 = thetahat[0],thetahat[1] # get parameters theta1, theta2 as their own variables
resid = curve_dict[2].get('fvec')
predicted = MichaelisMenton(puromycinx, t1, t2)
sqresid = []
for i in range(len(predicted)):
    sqresid.append((puromyciny[i]-predicted[i])**2)
ss = np.sum(sqresid)
n = len(puromycinx)
k = len(thetahat)
SSE = np.sqrt(ss/(n-k))
talpha2 = st.t.ppf(1-0.025,n-k)
x=0.4 # define the point of interest
dmmt1 = dmm_t1(x, t2) # calculate d/dtheta1
dmmt2 = dmm_t2(x, t1, t2) # calculate d/dtheta2
dv_vec = np.array([dmmt1,dmmt2]).T # derivative vector



varB = dv_vec.T @ varcov_mm @ dv_vec
sdB = np.sqrt(varB)
predsdB = np.sqrt(varB+1)
CIwidth = sdB * talpha2 * np.sqrt(SSE)
Predband = predsdB * talpha2

import uncertainties as unc
a,b = unc.correlated

yhat = MichaelisMenton(x, t1, t2)

yhat+CIwidth
yhat+Predband

print(varcov_mm)
np.diagonal(varcov_mm)


dv_vec @ np.linalg.inv(varcov_mm)
dv_vec @ np.linalg.inv(np.matmul(varcov_mm,varcov_mm.T))







linparamsstuff[0].get_prediction().summary_frame()
newfig = figure(height=500,width=800)
newfig.line(x=tresolved_countsdata_outlierremoved['Time']-min(tresolved_countsdata_outlierremoved['Time']),y=firstorder_LCI)
show(newfig)

fig,ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(tresolved_countsdata_outlierremoved['Time'],tresolved_countsdata_outlierremoved['206Pb/238U'],'-',color='k',lw=0.5,ms=0,label='Observed Data')
ax.plot(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[4],'-',color='blue',lw=0.5,ms=0,label='1st Order Regression')
ax.plot(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[5],'-',color='mediumpurple',lw=0.5,ms=0,label='Exp Regression')
ax.plot(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[0],'*',mfc='blue',ms=10,label='1st Order Intercept')
ax.plot(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[1],'*',mfc='mediumpurple',ms=10,label='Exp Intercept')
ax.errorbar(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[0],
            yerr=regression_uncertaintiesstuff[0]/100*regression_predictedstuff[0],lw=0.5,fmt='none',color='k'
            )
ax.errorbar(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[0],
            yerr=regression_uncertaintiesstuff[1]/100*regression_predictedstuff[1],lw=0.5,fmt='none',color='mediumpurple'
            )
    

fig,ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(tresolved_countsdata_outlierremoved['Time'],tresolved_countsdata_outlierremoved['207Pb/235U'],'-',color='k',lw=0.5,ms=0,label='Observed Data')
ax.plot(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[6],'-',color='blue',lw=0.5,ms=0,label='1st Order Regression')
ax.plot(tresolved_countsdata_outlierremoved['Time'], regression_predictedstuff[7],'-',color='mediumpurple',lw=0.5,ms=0,label='Exp Regression')
ax.plot(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[2],'*',mfc='blue',ms=10,label='1st Order Intercept')
ax.plot(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[3],'*',mfc='mediumpurple',ms=10,label='Exp Intercept')
ax.errorbar(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[2],
            yerr=regression_uncertaintiesstuff[2]/100*regression_predictedstuff[2],lw=0.5,fmt='none',color='k'
            )
ax.errorbar(tresolved_countsdata_outlierremoved.loc[0,'Time'], regression_predictedstuff[3],
            yerr=regression_uncertaintiesstuff[3]/100*regression_predictedstuff[3],lw=0.5,fmt='none',color='mediumpurple'
            )





def get_ellipse(data,power,ablation_start_true,regression_buttons,counts_mode):
    
    data = data.dropna()
    drop_condn = data[(data['206Pb/238U'] == 0) | (data['207Pb/235U'] == 0) | (data['207Pb/206Pb'] == 0)].index
    data.drop(drop_condn,inplace=True)
    data = data.reset_index(drop=True)
    
    predicted,predicted_207 = get_regressions(data,regression_buttons,ablation_start_true)
    if ('1st Order' in regression_buttons) and ('Exp. Regression' not in regression_buttons):
        data['207Pb/235U'] = predicted_207[0]
        data['206Pb/238U'] = predicted[0]
    else:
        data['207Pb/235U'] = predicted_207[1]
        data['206Pb/238U'] = predicted[1]
    
    x1 = data['207Pb/235U']
    y1 = data['206Pb/238U']
    x2 = 1/data['206Pb/238U']
    y2 = data['207Pb/206Pb']
    
    cov1 = np.cov(x1,y1) # get the 2x2 covariance matrix
    cov2 = np.cov(x2,y2)
    eigval1,eigvec1 = np.linalg.eig(cov1)  # get the eigen values and vectors
    eigval2,eigvec2 = np.linalg.eig(cov2)
    order1 = eigval1.argsort()[::-1] # arrange the indices in descending order
    order2 = eigval2.argsort()[::-1]
    eigvals_order1 = eigval1[order1] # order the eigval and eigvec by descending value
    eigvals_order2 = eigval2[order2]
    eigvecs_order1 = eigvec1[:,order1]
    eigvecs_order2 = eigvec2[:,order2]
    
    c1 = (np.mean(x1),np.mean(y1)) # center of data
    c2 = (np.mean(x2),np.mean(y2))
    wid1 = 2*np.sqrt(st.chi2.ppf((1-power),df=2)*eigvals_order1[0]) # get axes widths through bivariate normal
    hgt1 = 2*np.sqrt(st.chi2.ppf((1-power),df=2)*eigvals_order1[1])
    wid2 = 2*np.sqrt(st.chi2.ppf((1-power),df=2)*eigvals_order2[0])
    hgt2 = 2*np.sqrt(st.chi2.ppf((1-power),df=2)*eigvals_order2[1])
    theta1 = np.degrees(np.arctan2(*eigvecs_order1[:,0][::-1])) # get angle, converted from radians to degrees
    theta2 = np.degrees(np.arctan2(*eigvecs_order2[:,0][::-1]))
    
    ell1_params = [c1,wid1,hgt1,theta1] # put in list
    ell2_params = [c2,wid2,hgt2,theta2]
    
    return ell1_params,ell2_params













def threesigoutlierremoval(data,variable):
    threesig = 3*np.std(data[variable])
    mean = np.mean(data[variable])
    trigger = True
    while trigger is True:
        mask = (data.loc[:,variable] < mean - threesig) | (data.loc[:,variable] > mean + threesig)
        data.loc[mask,variable] = np.nan
        data.loc[:,variable] = data.loc[:,variable].infer_objects(copy=False).interpolate(method='linear')
        threesig = 3*np.std(data[variable])
        mean = np.mean(data[variable])
        if any(mask):
            pass
        else:
            break
               
    return data[variable]


def backgroundsubtract_convert_lod(data,bstart,bend,tstart,tend,arrayofdwelltimes):
    """
    Function for getting background subtracted counts data. 
    Takes raw intensities in cps, background subtracts the intensities, then converts to counts based on the integration time
    
    Parameters
    ----------
    data : pandas dataframe
        dataframe of time resolved intensities
    bstart : float
        user selected value for the background interval start
    bend : float
        user selected value for the background interval end
    tstart : float
        user selected value for the ablation interval start
    tend : float
        user selected value for the ablation interval end
    integrationtime : float
        integration time - stripped from the input file

    Returns
    -------
    ablation_backsub_counts : pandas dataframe
        pandas dataframe of background subtracted data converted into counts

    """
    backgrounds = data[(data['Time']>=bstart) & (data['Time']<=bend)]
    ablation = data[(data['Time']>=tstart) & (data['Time']<=tend)]
    
    for analyte in backgrounds.columns[1:]:
        outlier_removed_background = threesigoutlierremoval(backgrounds,analyte)
        outlier_removed_ablation = threesigoutlierremoval(ablation,analyte)
        backgrounds.loc[:,analyte] = outlier_removed_background
        ablation.loc[:,analyte] = outlier_removed_ablation
        

    meanbackgrounds = backgrounds.mean()[1:]
    lods = 3*backgrounds.std()[1:]

    ablation_backsub = ablation.iloc[:,1:].sub(meanbackgrounds,axis='columns').clip(lower=0)
    ablation_backsub_counts = ablation_backsub.loc[:,'238U':'202Hg']
    ablation_backsub_counts.insert(0,'Time',ablation['Time'])
    # ablation_backsub_counts = ablation_backsub_counts.reset_index(drop=True)
    
    return ablation_backsub_counts,meanbackgrounds,lods

dwellarray = np.full_like(np.arange(len(data.columns[1:]),dtype=float),0.01)

newdata,meanbackgrounds,lods = backgroundsubtract_convert_lod(data, bstart, bend, tstart, tend, dwellarray)






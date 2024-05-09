#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:15:39 2023

@author: Chuck Lewis, Oregon State University
"""

# %%
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import bokeh
from bokeh.plotting import figure
import holoviews as hv
import panel as pn
import param
import sys
import scipy
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('agg')
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

color_palette = bokeh.palettes.Muted9
color_palette_regressions = bokeh.palettes.Dark2_3
markers = ['o','d','^','v']
hv.extension('bokeh')

# %%
# define constants for reducing U-Pb data
lambda_238 = 1.55125e-10 # Jaffey et al. 1971
lambda_235 = 9.8485e-10 # Jaffey et al. 1971
lambda_232 = 4.9475e-11 # Le Roux and Glendenin 1963 / Steiger and Jager 1977
lambda_230 = 9.158e-6 # Cheng et al. 2000
# Errors on uraniumn decay constants from Mattionson(1987)
lambda_238_2sig_percent = 0.16 
lambda_235_2sig_percent = 0.21

SK74_2sig = 0.3
SK64_2sig = 1

# values from Woodhead and Hergt, 2001
pb_bias_dict = {'NIST-610':{'206Pb/204Pb': 17.047,'207Pb/204Pb': 15.509,'208Pb/204Pb': 36.975,'207Pb/206Pb': 0.9098},
                'NIST-612':{'206Pb/204Pb': 17.094,'207Pb/204Pb': 15.510,'208Pb/204Pb': 37.000,'207Pb/206Pb': 0.9073},
                'NIST-614':{'206Pb/204Pb': 17.833,'207Pb/204Pb': 15.533,'208Pb/204Pb': 37.472,'207Pb/206Pb': 0.8710}
                }
# values from Duffin et al., 2015
u_bias_dict = {'NIST-610':{'238U/235U': 419.4992},
               'NIST-612':{'238U/235U': 418.2650},
               'NIST-614':{'238U/235U': 374.4964}
               }
# 'true' isotope masses
mass_dict = {'238U': 238.050788427,
             '235U': 235.043929918,
             '232Th': 232.038055325,
             '208Pb': 207.976652071,
             '207Pb': 206.975896887,
             '206Pb': 205.974465278,
             '204Pb': 203.973043589,
             '202Hg': 201.970643011,
             }
# create a dictionary that holds known or estimated U/Th ratios of zircon and associated magma for standards, as well as common Pb ratios
stds_dict = {'Temora': [2.4, 0.79200, 18.0528, 15.5941, 0.055137],  # Black et al., 2004. Ambiguous Th correction > assumed D = 0.33
              # Schmitz and Bowring 2001; only one with measured common Pb so far
              'FishCanyon': [1.496558, 0.454545, 18.4275, 15.5425, 0.046615],
              # Klepeis et al 1998
              '94-35': [1, 0.33, 18.6191, 15.626, 0.047145],
              # Slama et al 2008
              'Plesovice': [10.7, 0.25, 18.1804, 15.6022, 0.053218],
              # Black et al., 2004. Ambiguous Th correction > assumed D = 0.33
              'R33': [1.4, 0.46200, 18.0487, 15.5939, 0.055199],
              # Wiedenbeck et al 1995
              '91500': [1, 0.33, 16.9583, 15.4995, 0.074806],
              # Paces and Miller 1993. Ambiguous Th correction > assumed D = 0.33
              'FC1': [1.7, 0.56100, 16.892, 15.492, 0.076203],
              # unpublished; Bowring > assumed D = 0.33
              'Oracle': [2.2, 0.725999, 16.2726, 15.4099, 0.090545],
              # Pecha unpublished > assumed D = 0.33
              'Tan-Bra': [1.2, 0.39600, 14.0716, 14.8653, 0.165098],
              # Stern et al 2009. Ambiguous Th correction > assumed D = 0.33
              'OG1': [1.3, 0.42900, 11.8337, 13.6071, 0.294475]
              }  

accepted_ages = {
    'Temora': [416780000,416.94*1e6],
    'FishCanyon': [28478000,28.528*1e6],
    '94-35': [55500000,55.07*1e6],
    'Plesovice': [337100000,337.26*1e6],
    'R33': [419300000,418.41*1e6],
    '91500': [1062400000,1063501210],
    'FC1': 1099500000,
    'Oracle': 1436200000,
    'Tan-Bra': 2507800000,
    'OG1': 3440700000
}

TIMS_errors = {
    'Temora': [330000,0.0566*1e6],
    'FishCanyon': [24000,0.03*1e6],
    '94-35': [80000,0.3*1e6],
    'Plesovice': [200000,0.011*1e6],
    'R33': [400000,0.105*1e6],
    '91500': [1900000,811981],
    'FC1': [330000,330000],
    'Oracle': [1300000,1300000],
    'Tan-Bra': [1500000,1500000],
    'OG1': [3200000,0.777*1e6]
}


# set up a map that maps variables to a dropdown of labels that are easier to read and understand
drift_variable_map = {'206Pb/238U Age': ['206Pb/238U_correctedage','206/238U_age_init'],'207Pb/235U Age': ['207Pb/235Uc_age','207/235U_age_init'],
                      '206Pb/238U': ['206Pb/238Upbc','206Pb/238U_unc'],'207Pb/235U': ['207Pb/235U_corrected','207Pb/235U'],'238U/235U': ['238U/235U c','238U/235U'],
                      '207Pb/206Pb: By Measured Mass Bias': ['207Pb/206Pb c','207Pb/206Pb'],
                      '207Pb/206Pb: By Age': ['207Pb/206Pbr','207Pb/206Pb']
                      }

# %%


class calc_fncs:
    """ Class that holds all of the functions for reducing the reduced LAICPMS data"""

    def __init__(self, *args):
        for a in args:
            self.__setattr__(str(a), args[0])
        

    def get_data_TW_regressions(df,regression_var,common_207206_input,callingmethod):
        """
        Function that regresses data from common Pb to Tera-Wasserburg concordia

        Parameters
        ----------
        df : pandas dataframe
            hosts measured data.
        regression_var : string
            determines what 206/238 ratio should be used to project - measured or mass bias corrected. 
            i.e., when calculating ages do we use the measured or concordant age?
        common_207206_input : float
            value that has a common Pb 207/206 ratio. Zero if none input by user
        callingmethod : string
            which method called this method. Values going into regression calculation changes based on input:
                if correct_standard_ages, the SK common Pb variable is assigned in the correct standard ages function to be accepted values in dictionaries
                if anything else, checks if the common Pb was manually input and regresses based on that value

        Returns
        -------
        discordia_t : list
            contains regression parameters for each data point.

        """
        if callingmethod == 'correct_standard_ages':
            pts_x = np.array([df[regression_var], np.zeros_like(df['SK 207Pb/206Pb'])]).T # set up an array the length of the dataframe containing the corresponding 38/6 ratios
            pts_y = np.array([df['207Pb/206Pb c'], df['SK 207Pb/206Pb']]).T # set up an array the length of hte dataframe containing the corresponding 7/6 ratios
            discordia_t = np.zeros((len(df), 2)) # set up an array to be filled with regression parameters in the first order
            # loop through and calculate regression parameters for each point
            for i in range(0, len(df)):
                discordia_t[i] = np.poly1d(np.polyfit(pts_x[i], pts_y[i], 1))
        
        else:
            pts_x = np.array([df[regression_var], np.zeros_like(df['SK 207Pb/206Pb'])]).T
            if common_207206_input == 0:
                pts_y = np.array([df['207Pb/206Pb c'], df['SK 207Pb/206Pb']]).T
            else:
                pts_y = np.array([df['207Pb/206Pb c'], df['Common 207Pb/206Pb']]).T
            discordia_t = np.zeros((len(df), 2))
            
            for i in range(0, len(df)):
                discordia_t[i] = np.poly1d(np.polyfit(pts_x[i], pts_y[i], 1))
 

        return discordia_t

    def get_TW_concordia():
        """
        Calculates Tera-Wasserburg Concordia and returns the X-Y values corresponding to concordia in two arrays

        Returns
        -------
        u238_pb206 : array
            x values to TW concordia
        pb207_206r : array
            y values to TW concordia

        """
        t = np.linspace(1, 4.6e9, 100000) # set up array across earth time
        u238_pb206 = np.zeros(len(t)) # set up array of zeros to be filled with x vals
        pb207_206r = np.zeros(len(t)) # same but for y vals
        # loop through and assign values based on age equation
        for i in range(0, len(t)):
            u238_pb206[i] = 1/(np.exp(lambda_238*t[i])-1)
            pb207_206r[i] = (1/137.818) * ((np.exp(lambda_235*t[i])-1) / (np.exp(lambda_238*t[i])-1))

        return u238_pb206, pb207_206r
    
    def get_Weth_concordia():
        """
        Calculates Wetherhill Concordia and returns the X-Y values corresponding to concordia in two arrays

        Returns
        -------
        pb207_u235 : array
            X values to Wetherhill concordia
        pb206_u238 : array
            Y values to wetherhill concordia

        """
        t = np.linspace(1, 4.6e9, 100000) # set up array across earth time
        pb206_u238 = np.zeros(len(t)) # same but for y vals
        pb207_u235 = np.zeros(len(t)) # set up array of zeros to be filled with x vals
        # loop through and assign values based on age equation
        for i in range(0, len(t)):
            pb206_u238[i] = np.exp(lambda_238*t[i])-1
            pb207_u235[i] = np.exp(lambda_235*t[i])-1

        return pb207_u235,pb206_u238

    def get_projections(df, ellipse_mode_selector, power,common_207206_input):
        """
        Function that gets the projected concordia values using the regressions onto Tera-Wasserburg concordia

        Parameters
        ----------
        df : pandas dataframe
            hosts measured and reduced data
        ellipse_mode_selector : string
            Used for user to select if they want point estimates, ellipsoids, or both
                At present does nothing in this function, but will in a futre update
        power : float
            power values for ellipsoid
                At present does nothing in this function, but will in a futre update
        common_207206_input : float
            user input value for common Pb correction. Zero if none input

        Returns
        -------
        points : array
            array of points along concordia that correspond to projections
            if there is an analysis that has a projection that passes through concordia twice, the younger age is passed over to the concordant ratios
            used for plotting otherwise
        concordia_238_206 : array
            concordant 238/206 ratios
            used for age calculations
        pts_pb_r : array
            concordant 207/206 ratios (i.e., the radiogenic lead ratio)
            used for age calculations
        discordia_207206 : array
            input 207/206 ratio for the function
            used to check work and not removed from function for ease in future updates
        discordia_238206 : array
            input 207/206 ratio for the function
            used to check work and not removed from function for ease in future updates

        """
        callingmethod = sys._getframe().f_back.f_code.co_name
        # print('Projections Calling Method'+str(callingmethod))
        #     regression_var = '238U/206Pb_corrected'
        if callingmethod == 'get_pt_ages':
            regression_var = '238U/206Pb_corrected'
        elif callingmethod == 'get_ellipse_ages':
            regression_var = '238U/206Pb_corrected'
        else:
            regression_var = '238U/206Pb'
        # get the TW concordia values
        x_TW, y_TW = calc_fncs.get_TW_concordia()
        discorida_regressions = calc_fncs.get_data_TW_regressions(df,regression_var,common_207206_input,callingmethod)  # get regressions
        # array of xvalues to project over
        x_vals = np.linspace(min(x_TW), max(x_TW), 100000)
        # set up array to be filled with calculated radiogenic lead component
        pts_pb_r = np.zeros(len(discorida_regressions))
        concordia_238_206 = np.zeros(len(discorida_regressions))

        for i in range(0, len(discorida_regressions)):

            discordia_207206 = discorida_regressions[i][0] * x_vals+discorida_regressions[i][1]
            discordia_238206 = (discordia_207206-discorida_regressions[i][1])/discorida_regressions[i][0]

            # distance of y value from line
            delta_y = (discorida_regressions[i][1] +x_TW * discorida_regressions[i][0]) - y_TW
            # index the regression for where the curve cross the regression
            indx = np.where(delta_y[1:]*delta_y[:-1] < 0)[0]
            # similar triangles geometry gets points
            d_ratio = delta_y[indx] / (delta_y[indx] - delta_y[indx + 1])
            # empty array for crossing points
            points = np.zeros((len(indx), 2))
            points[:, 0] = x_TW[indx] + d_ratio * (x_TW[indx+1] - x_TW[indx])  # x crossings
            points[:, 1] = y_TW[indx] + d_ratio * (y_TW[indx+1] - y_TW[indx])  # y crossings
            y_point = y_TW[indx] + d_ratio * (y_TW[indx+1] - y_TW[indx])  # y crossings
            x_point = x_TW[indx] + d_ratio * (x_TW[indx+1] - x_TW[indx])  # x crossings
            # check if more than one point exists. If so, give the younger concordant projection
            if len(y_point) >= 1:
                pts_pb_r[i] = min(y_point)
            elif len(y_point) < 0:
                pts_pb_r[i] = 0

            if len(x_point) > 1:
                concordia_238_206[i] = min(x_point)
            elif len(x_point) == 1:
                concordia_238_206[i] = x_point

        return points, concordia_238_206, pts_pb_r, discordia_207206, discordia_238206

    def get_ellipse(df, power):
        """
        Function that calculates ellipsoid parameters

        Parameters
        ----------
        df : pandas dataframe
            hosts measured data from integrations in the ablation
        power : float
            alpha value for confidence

        Returns
        -------
        ell2_params : array
            hosts parameters for ellipsoid on Tera-Wasserburg concordia
            format: (centerx,centery),width,height,angle
        ellw_params : array
            same but for Wetherhill concordia

        """
        # get which method called this one. If get_allipse_ages, use mass bias corrected values for Pb-U ratios. Otherwise use measured values (for plotting)
        # 7/6 ratio used depends on how mass bias was calculated
        callingmethod = sys._getframe().f_back.f_code.co_name
        if callingmethod == 'get_ellipse_ages':
            x2 = 1/df['206Pb/238U_corrected']
            xw = df['207Pb/235U_corrected']

        else:
            x2 = df['238U/206Pb']
            xw = df['207Pb/235U']
        
        # pass zeros if there is nothing so function doesn't break program
        if df is None:
            ell2_params = [(0,0), 1, 1, 0]
            ellw_params = [(0,0), 1, 1, 0]
        else:
            y2 = df['207Pb/206Pb c']
            yw = df['206Pb/238U']
    
            cov2 = np.cov(x2, y2) # get covariance of x and y data
            covw = np.cov(xw,yw)
            eigval2, eigvec2 = np.linalg.eig(cov2) # get eigen value, vector from the covariance matrix
            eigvalw, eigvecw = np.linalg.eig(covw)
            order2 = eigval2.argsort()[::-1] # sort the eigen values
            orderw = eigvalw.argsort()[::-1]
            eigvals_order2 = eigval2[order2] # apply the sorted indices into a new variable that hosts sorted values
            eigvals_orderw = eigvalw[orderw]
            eigvecs_order2 = eigvec2[:, order2]
            eigvecs_orderw = eigvecw[:, orderw]
    
            c2 = (np.mean(x2), np.mean(y2)) # get center of ellipsoid
            cw = (np.mean(xw), np.mean(yw))
            wid2 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power), df=2) * eigvals_order2[0]) # get the width
            widw = 2*np.sqrt(scipy.stats.chi2.ppf((1-power), df=2) * eigvals_orderw[0])
            hgt2 = 2*np.sqrt(scipy.stats.chi2.ppf((1-power), df=2)* eigvals_order2[1]) # get the height
            hgtw = 2*np.sqrt(scipy.stats.chi2.ppf((1-power), df=2)* eigvals_orderw[1])
            theta2 = np.degrees(np.arctan2(*eigvecs_order2[:, 0][::-1])) # get the angle
            thetaw = np.degrees(np.arctan2(*eigvecs_orderw[:, 0][::-1]))
    
            ell2_params = [c2, wid2, hgt2, theta2] # put values in an array to be returned
            ellw_params = [cw, widw, hgtw, thetaw]

        return ell2_params,ellw_params

    def plot_TW(df, ell_df, x_axis_range, y_axis_range, label_toggle, ellipse_mode_selector, power, common_207206_input):
        """
        Function that plots Tera-Wasserburg concordia

        Parameters
        ----------
        df : pandas dataframe
            has measured data
        ell_df : pandas dataframe
            has measured ellipsoid data.
        x_axis_range : array
            float values determining the x-axis range
        y_axis_range : aray
            float values determining the y-axis range
        label_toggle : string
            determines whether or not to plot sample names next to their points
            currently needs update
        ellipse_mode_selector : string
            Used for user to select if they want point estimates, ellipsoids, or both
        power : float
            alpha value for ellipsoid.
        common_207206_input : float
            user input value for common Pb correction. Zero if none input

        Returns
        -------
        fig : matplotlib figure
            has TW concordia, measured points, projected points, projections.

        """
        df = df.reset_index(drop=True) # reset indices
        # create figure
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot()
        x_TW, y_TW = calc_fncs.get_TW_concordia() # get concordia curve
        
        # in terms of coding it is ugly to do this here too, but it is definitely faster than looping through the get_projections method.. can't have it all
        regression_var = '238U/206Pb' # set the regression variable to be measured value
        disc_regressions = calc_fncs.get_data_TW_regressions(df,regression_var,common_207206_input,'plot_TW') # get projected points
        
        
        # get and plot projections if point estimates desired by user
        if (ellipse_mode_selector == 'Point Estimates' or ellipse_mode_selector == 'Both'):
            x_vals = np.linspace(min(x_TW), max(x_TW), 100000) # set up xvals for regression
            pts_pb_r = np.zeros(len(disc_regressions)) # set array of zeros to be filled with radiogenic lead points
            
            for i in range(0, len(disc_regressions)):
                discordia_207206 = disc_regressions[i][0] * x_vals+disc_regressions[i][1]
                discordia_238206 = (discordia_207206-disc_regressions[i][1])/disc_regressions[i][0]
    
                # distance of y value from line
                delta_y = (disc_regressions[i][1] + x_TW * disc_regressions[i][0]) - y_TW
                # index the regression for where the curve cross the regression
                indx = np.where(delta_y[1:]*delta_y[:-1] < 0)[0]
                # similar triangles geometry gets points
                d_ratio = delta_y[indx] / (delta_y[indx] - delta_y[indx + 1])
                # empty array for crossing points
                points = np.zeros((len(indx), 2))
                points[:, 0] = x_TW[indx] + d_ratio * (x_TW[indx+1] - x_TW[indx])  # x crossings
                points[:, 1] = y_TW[indx] + d_ratio * (y_TW[indx+1] - y_TW[indx])  # y crossings
                y_point = y_TW[indx] + d_ratio * (y_TW[indx+1] - y_TW[indx])  # y crossings
                if len(y_point) >= 1:
                    pts_pb_r[i] = min(y_point)
                elif len(y_point) < 0:
                    pts_pb_r[i] = 0
                ax.plot(discordia_238206, discordia_207206, '-k', lw=0.5)
                ax.plot(points[:, 0], points[:, 1], 'o', mfc='darkkhaki', mec='k')
        
        if ellipse_mode_selector == 'Point Estimates':
            ax.plot(x_TW, y_TW, 'k', lw=1)
            ax.errorbar(df['238U/206Pb'], df['207Pb/206Pb c'], xerr=df['238/206 err'],yerr=df['SE 207/206'], fmt='none', ecolor='k', elinewidth=0.5)
            ax.plot(df['238U/206Pb'], df['207Pb/206Pb'], 'd', mfc='yellow', mec='k')
            if 'Concordia' in label_toggle:
                for x, y, t in zip(df['238U/206Pb'], df['207Pb/206Pb'], df['SampleLabel']):
                    label = t
                    ax.annotate(label, (x, y),textcoords="offset points",  xytext=(0, 10),ha='center',fontsize=8)
            else:
                pass
        elif ellipse_mode_selector == 'Ellipses':
            ell_df = ell_df.reset_index(drop=True)
            ax.plot(x_TW, y_TW, 'k', lw=1)
            for i in ell_df.SampleLabel.unique():
                conf_ellipse,conf_ellipseW = calc_fncs.get_ellipse(ell_df[ell_df['SampleLabel'] == i], power)
                ell = Ellipse(xy=conf_ellipse[0],width=conf_ellipse[1],height=conf_ellipse[2],angle=conf_ellipse[3],color='darkgray',ec='k',alpha=0.5) # set the parameters into a plotable 'patch'
                ax.add_artist(ell)
            
        elif ellipse_mode_selector == 'Both':
            ax.plot(x_TW, y_TW, 'k', lw=1)
            ell_df = ell_df.reset_index(drop=True)
            for i in ell_df.SampleLabel.unique():
                conf_ellipse,conf_ellipseW = calc_fncs.get_ellipse(ell_df[ell_df['SampleLabel'] == i], power)
                ell = Ellipse(xy=conf_ellipse[0],width=conf_ellipse[1],height=conf_ellipse[2],angle=conf_ellipse[3],color='darkgray',ec='k',alpha=0.5) # set the parameters into a plotable 'patch'
                ax.add_artist(ell)
            ax.plot(x_TW, y_TW, 'k', lw=1)
            ax.errorbar(df['238U/206Pb'], df['207Pb/206Pb c'], xerr=df['238/206 err'],yerr=df['SE 207/206'], fmt='none', ecolor='k', elinewidth=0.5)
            ax.plot(df['238U/206Pb'], df['207Pb/206Pb'], 'd', mfc='yellow', mec='k')
            if 'Concordia' in label_toggle:
                for x, y, t in zip(df['238U/206Pb'], df['207Pb/206Pb'], df['SampleLabel']):
                    label = t
                    ax.annotate(label, (x, y),textcoords="offset points",  xytext=(0, 10),ha='center',fontsize=8)
            else:
                pass
        
        ax.set_xlim(x_axis_range)
        ax.set_ylim(y_axis_range)
        ax.set_ylabel('207Pb/206Pb', fontsize=6) # set ylabel
        ax.set_xlabel('238U/206Pb', fontsize=6) # blank x label
        ax.tick_params(axis='both', labelsize=5) # put ticks on axes
        return fig



    def plot_weth(df, ell_df, x_axis_range, y_axis_range, label_toggle, ellipse_mode_selector, power):
        """
        Function that plots Wetherhill concordia

        Parameters
        ----------
        df : pandas dataframe
            measured and reduced data
        ell_df : pandas dataframe
            measured and reduced data
        x_axis_range : array
            float values determining the x-axis range
        y_axis_range : aray
            float values determining the y-axis range
        label_toggle : string
            determines whether or not to plot sample names next to their points
            currently needs update
        ellipse_mode_selector : string
            Used for user to select if they want point estimates, ellipsoids, or both
        power : float
            alpha value for ellipsoid.

        Returns
        -------
        fig : matplotlib figure
            Wetherhill concordia with points ± ellipsoids and concordia curve

        """
        df = df.reset_index(drop=True) # reset indices in dataframe
        # initialize figure, set plot aesthetics
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot()
        # get concordia curve
        x_Weth, y_Weth = calc_fncs.get_Weth_concordia()            

        if ellipse_mode_selector == 'Point Estimates':
            ax.plot(x_Weth, y_Weth, 'k', lw=1)
            ax.errorbar(df['207Pb/235U'], 1/df['238U/206Pb'], xerr=df['207/235 Reg. err'],yerr=df['206/238 Reg. err'], fmt='none', ecolor='k', elinewidth=0.5)
            ax.plot(df['207Pb/235U'], 1/df['238U/206Pb'], 'd', mfc='yellow', mec='k')
            if 'Concordia' in label_toggle:
                for x, y, t in zip(df['207Pb/235U'], 1/df['238U/206Pb'], df['SampleLabel']):
                    label = t
                    ax.annotate(label, (x, y),textcoords="offset points",  xytext=(0, 10),ha='center',fontsize=8)
            else:
                pass
            
        elif ellipse_mode_selector == 'Ellipses':
            
            ell_df = ell_df.reset_index(drop=True)
            ax.plot(x_Weth, y_Weth, 'k', lw=1)
            for i in ell_df.SampleLabel.unique():
                conf_ellipse,conf_ellipseW = calc_fncs.get_ellipse(ell_df[ell_df['SampleLabel'] == i], power)
                ell = Ellipse(xy=conf_ellipseW[0],width=conf_ellipseW[1],height=conf_ellipseW[2],angle=conf_ellipseW[3],color='darkgray',ec='k',alpha=0.5) # set the parameters into a plotable 'patch'
                ax.add_artist(ell)
        
        elif ellipse_mode_selector == 'Both':
            ell_df = ell_df.reset_index(drop=True)
            ax.plot(x_Weth, y_Weth, 'k', lw=1)
            for i in ell_df.SampleLabel.unique():
                conf_ellipse,conf_ellipseW = calc_fncs.get_ellipse(ell_df[ell_df['SampleLabel'] == i], power)
                ell = Ellipse(xy=conf_ellipseW[0],width=conf_ellipseW[1],height=conf_ellipseW[2],angle=conf_ellipseW[3],color='darkgray',ec='k',alpha=0.5) # set the parameters into a plotable 'patch'
                ax.add_artist(ell)
            ax.plot(x_Weth, y_Weth, 'k', lw=1)
            ax.errorbar(df['207Pb/235U'], 1/df['238U/206Pb'], xerr=df['207/235 Reg. err'],yerr=df['206/238 Reg. err'], fmt='none', ecolor='k', elinewidth=0.5)
            ax.plot(df['207Pb/235U'], 1/df['238U/206Pb'], 'd', mfc='yellow', mec='k')
            if 'Concordia' in label_toggle:
                for x, y, t in zip(df['207Pb/235U'], 1/df['238U/206Pb'], df['SampleLabel']):
                    label = t
                    ax.annotate(label, (x, y),textcoords="offset points",  xytext=(0, 10),ha='center',fontsize=8)
            else:
                pass
        ax.set_xlim(x_axis_range)
        ax.set_ylim(y_axis_range)
        ax.set_ylabel('206Pb/238U', fontsize=6) # set ylabel
        ax.set_xlabel('207Pb/235U', fontsize=6) # blank x label
        ax.tick_params(axis='both', labelsize=5) # put ticks on axes
        return fig
    
    

    def plot_boxplot(ages, analysis_ID, label_toggle, ellipse_mode_selector):
        """
        Function that creates a boxplot for the requested data

        Parameters
        ----------
        ages : pandas dataframe
            df with calculated ages
        analysis_ID : pandas series
            sample labels for each analysis
        label_toggle : list
            contains strings that determine which plot to include sample labels on.
        ellipse_mode_selector : string
            allows user to choose if points estimates, ellipsoids, or both data types should be included in reduction.

        Returns
        -------
        Depndends on if user requests point estimates in reductions. If yes:
            matplotlib figure with boxplot
        if no:
            prints string

        """
        # check if point estimates are requested by user
        if ellipse_mode_selector == 'Point Estimates':
            fig = Figure(figsize=(2, 4)) # initialize figure
            ax = fig.add_subplot() # add axis to figure
            # set up box plot with the data
            bp = ax.boxplot(ages, patch_artist=True, boxprops=dict(facecolor='darkgray', color='k'),
                            medianprops=dict(color='yellow'), meanprops=dict(marker='d', mfc='yellow', mec='k', markersize=4),
                            flierprops=dict(
                                marker='o', mfc='None', mec='k', markersize=4),
                            showmeans=True)
            # put text on plot for mean, median, min, max, and n. Set location to be relative to axis dimensions (not data dimensions)
            ax.text(0.05, 0.8, 'Mean ='+str(round(ages.mean(), 2)),
                    fontsize=4, transform=ax.transAxes)
            ax.text(0.05, 0.7, 'Med ='+str(round(ages.median(), 2)),
                    fontsize=4, transform=ax.transAxes)
            ax.text(0.05, 0.6, 'Min ='+str(round(ages.min(), 2)),
                    fontsize=4, transform=ax.transAxes)
            ax.text(0.05, 0.5, 'Max ='+str(round(ages.max(), 2)),
                    fontsize=4, transform=ax.transAxes)
            ax.text(0.05, 0.4, 'n = '+str(len(ages)),
                    fontsize=4, transform=ax.transAxes)
            
            ax.set_ylabel('206/238 Age (Ma)', fontsize=5) # set ylabel
            ax.set_xlabel(' ', fontsize=1) # blank x label
            ax.tick_params(axis='both', labelsize=4) # put ticks on axes
            # check if sample labels on box and whisker are requested. If so, plot sample labels next to outliers
            if 'Box + Whisker' in label_toggle:
                fliers = [item.get_ydata() for item in bp['fliers']]
                for x, t in zip(ages, analysis_ID):
                    if t in fliers:
                        label = t
                        age = x
                        ax.annotate(label, age, textcoords='offset points', ha='center', fontsize=5)

            return fig
        else:
            return print('In Ellipse Mode')

    def plot_drift(std_df, secondary_df, secondary_list, unknown_df, drift_var,
                   std_txt,ThU_zrn,ThU_magma,Pb_Th_std_crct_selector,regression_selector,ellipse_mode_selector,power,UTh_std_norm,common_207206_input,common_207206_error,
                   drift_treatment,drift_nearest):
        """
        Function that effectively plots the drift. Specifically, looks at the fractionation factor for the requested variable for all secondary standards

        Parameters
        ----------
        std_df : pandas dataframe
            df containing measured and secondary standard values for primary standard
        secondary_df : pandas dataframe
            df containing measured and secondary standard values for secondary standard
        secondary_list : list
            list of secondary standards. Used to look for standards in samplelabels of df
        unknown_df : pandas dataframe
            unknowns. Used for testing and currently is not used in the fucntion
        drift_var : string
            variable requested by user. Maps to dictionary
        std_txt : string
            used for testing. Currently serves no function
        ThU_zrn : Th/U ratio in zircon
            used for testing. Currently serves no function.
        ThU_magma : Th/U ratio in host melt/magma
            used for testing. Currently serves no function.
        Pb_Th_std_crct_selector : TYPE
            used for testing. Currently serves no function..
        regression_selector : TYPE
            used for testing. Currently serves no function..
        ellipse_mode_selector : TYPE
            used for testing. Currently serves no function..
        power : TYPE
            used for testing. Currently serves no function..
        UTh_std_norm : TYPE
            used for testing. Currently serves no function..
        common_207206_input : TYPE
            used for testing. Currently serves no function..
        common_207206_error : TYPE
            used for testing. Currently serves no function..
        drift_treatment : TYPE
            used for testing. Currently serves no function..
        drift_nearest : TYPE
            used for testing. Currently serves no function..

        Returns
        -------
        fig : bokeh figure
            has fractionation factors vs measurement number for standards

        """
        fig = figure(height=300,width=1000,title='Drift Assessment',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='right',
                     x_axis_label='Measurement #',y_axis_label=str(drift_var))
        fig.xgrid.grid_line_color = 'darkgray'
        fig.ygrid.grid_line_color = 'darkgray'
        drift_var_mapped = drift_variable_map.get(drift_var)[0]
        uncrct_drift_var_mapped = drift_variable_map.get(drift_var)[1]
        secondary_stds = secondary_list
        dummy_df = unknown_df
        for s in range(0,len(secondary_list)):
            secondary_std_s = secondary_df[secondary_df['Sample'] == secondary_stds[s]]
            secondary_std_s = secondary_std_s.reset_index(drop=True)
            secondary_std_s['fracfactor'] = secondary_std_s[drift_var_mapped]/secondary_std_s[uncrct_drift_var_mapped]
            xvals = secondary_std_s['measurementindex']
            yvals = secondary_std_s['fracfactor']
            fig.diamond(xvals,yvals,size=10,color=color_palette[s],legend_label=str(secondary_std_s['Sample'][0]))
        
        return fig

    def correct_standard_ages(df, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input):
        """
        Function used to calculate standard ages

        Parameters
        ----------
        df : pandas dataframe
            has measaured and reduced standard data
        std_txt : string
            standard name
        Pb_Th_std_crct_selector : string
            determines if only common Pb or Common Pb and Th disequilibrium should be corrected for
        regression_selector : string
            how Pb-U ratios were treated (exponential, total counts, 1st order)
        ellipse_mode_selector : string
            whether to use point estimates, ellipsoids, or both
        power : float
            alpha value in confidence ellipse
        common_207206_input : float
            user input value for common 7/6 in common Pb correction. Zero if non input

        Returns
        -------
        avg_std_age : float
            average age of standard.
        avg_std_age_Thcrct : float
            average age of standard with Th correction.
        avg_std_age_207 : float
            average 207/235 age of standard.
        avg_std_ratio : float
            average 206/238 ratio of standard.
        avg_std_ratio_Thcrct : float
            average 206/238 ratio of standard with Th correction.
        avg_std_ratio_207 : float
            average 207/235 ratio of standard.
        avg_reg_err : float
            average error on the 206/238 ratio.
        avg_reg_err_207 : float
            average error on the 207/235 ratio.
        UTh_std : float
            accepted U/Th ratio in standards
        UTh_std_m : float
            measured U/Th ratio in standards.
        fracfactor_76 : float
            fractionation factor on the 7/6 ratio.

        """
        df = df.reset_index(drop=True) # reset indices
        

        # set up empty arrays to be filled for common lead corrections
        common_filter = np.zeros(len(df))

        pb_m = df['207Pb/206Pb']  # measured 207/206
        if Pb_Th_std_crct_selector == 'Common Pb':
            df['SK 207Pb/206Pb'] = stds_dict.get(std_txt)[3] / stds_dict.get(std_txt)[2]
            common = df['SK 207Pb/206Pb']
        elif Pb_Th_std_crct_selector == 'Common Pb + Th Disequil.':
            # calculated Stacey-Kramers 207/206 overwriting the input 207/206 should manual values be requested
            df['SK 207Pb/206Pb'] = stds_dict.get(std_txt)[3] / stds_dict.get(std_txt)[2]
            common = df['SK 207Pb/206Pb']
        # get values projected onto concordia
        points, concordia_238_206, pts_pb_r, na, naII = calc_fncs.get_projections(df, ellipse_mode_selector, power,common_207206_input)
        # if common Pb not feasible, assign zero. Retain value otherwise
        for i in range(0,len(common)):
            if common[i] <= 0:
                common_filter[i] = 0
            else:
                common_filter[i] = common[i]
                

        # calculate fraction of common Pb
        f_ = (pb_m - pts_pb_r) / (common - pts_pb_r)
        # set up array to set f = 0 if point lies on or below Concordia (i.e., no common Pb present)
        f = np.zeros(len(common_filter))

        for k, j in zip(common_filter, range(0,len(f_))):
            if k <= 0:
                f[j] = 0
            elif f_[j] < 0:
                f[j] = 0
            else:
                f[j] = f_[j]

        # append the calculations to the sample dataframe
        df['207Pb/206Pbr'] = pts_pb_r
        # get the concordant 7/6 ratio for the accepted standard age
        std_concardant_76 = stds_dict.get(std_txt)[4]
        # get the fractionation factor on the 7/6 ratio from the standard measurements and the accepted values
        zrm_pb_f = np.log(std_concardant_76/np.mean(df['207Pb/206Pb']))/np.log(mass_dict.get('207Pb')/mass_dict.get('206Pb'))
        # reassign variable
        fracfactor_76 = zrm_pb_f
        # assign dataframe f
        df['f'] = f

        df['counts_pb206r'] = df['206Pb'] * (1-df['f']) # calculate counts of radiogenic 206
        df['206Pb/238Upbc_numerical'] = 1 / df['238U/206Pb']-(1/df['238U/206Pb']*f) # calculate 6/38 numerically to make sure all calculations follow theory. Used for testing
        df['206Pb/238Upbc'] = 1/concordia_238_206 # get concordant 6/38 ratio from projections
        df['206Pb/238Uc_age'] = np.log(df['206Pb/238Upbc'] + 1) / lambda_238 # calculate age of common Pb corrected ratio (concordant point)
        UThstd, UThstd_rx = stds_dict.get(std_txt)[0], stds_dict.get(std_txt)[1] # get the U/Th ratio of standards and their host rocks
        DThU = (1/UThstd)/(1/UThstd_rx) # get the D value
        df['206Pb/238UPbThc'] = df['206Pb/238Upbc'] - (lambda_238/lambda_230*(DThU-1)) # get the 6/38 ratio corrected for common Pb and Th disequil
        df['206Pb/238UPbTh_age'] = np.log(df['206Pb/238UPbThc'] + 1) / lambda_238 # calculate age of common Pb + Th disequil. ratio
        UTh_std_m = df['238U'].mean()/df['232Th'].mean() # Measured 38/32 ratio from standard
        
        df['207Pb/235Upbc'] = df['207Pb/235U']-(df['207Pb/235U']*f) # common Pb corrected 7/35 ratio (assumes f value is same from 6/38..)
        df['207Pb/235Uc_age'] = np.log(df['207Pb/235Upbc'] + 1) / lambda_235 # calculate 7/35 age from common Pb corrected ratio

        avg_std_age = df['206Pb/238Uc_age'].mean() # average common Pb corrected standard age
        avg_std_age_Thcrct = df['206Pb/238UPbTh_age'].mean() # average common Pb + Th disequil. corrected standard age
        avg_std_age_207 = df['207Pb/235Uc_age'].mean() # average common Pb corrected 7/35 standard age

        avg_std_ratio = 1/df['238U/206Pb'].mean() # average 6/38 ratio from standard
        avg_std_ratio_Thcrct = df['206Pb/238UPbThc'].mean() # average 6/38 ratio from standard, corrected for Common pb and Th disequil
        avg_std_ratio_207 = df['207Pb/235Upbc'].mean() # average 7/35 ratio from standard, corrected for common Pb
        
        # calculate weighted means and errors for the standards
        # initialize arrays to be populated
        wts = np.zeros(len(df))
        wts207 = np.zeros(len(df))
        wts = np.zeros(len(df))
        wts207 = np.zeros(len(df))
        xbarwtd_num = np.zeros(len(df))
        xbarwtd_num_207 = np.zeros(len(df))
        vwtd_num = np.zeros(len(df))
        vwtd_num_207 = np.zeros(len(df))
        if ellipse_mode_selector == 'Point Estimates':
            for i in range(0, len(df)):
                wt_se_i = 1/df['206/238 Reg. err'][i] # weight of analysis
                xi = df['206Pb/238Upbc'][i] # observed value
                xi2 = df['206Pb/238Upbc'][i]**2 # observed value squared
                xbarwtd_num_i = wt_se_i*xi # numerator of weighted mean
                vwtd_num_i = wt_se_i*xi2 # numerator of weighted variance
                # populate arrays
                wts[i] = wt_se_i 
                xbarwtd_num[i] = xbarwtd_num_i
                vwtd_num[i] = vwtd_num_i
                # same as above but for 7/35
                wt_se_i_207 = 1/df['207/235 Reg. err'][i]
                xi_207 = df['207Pb/235Uc_age'][i]
                xi2_207 = df['207Pb/235Uc_age'][i]**2
                xbarwtd_num_i_207 = wt_se_i_207*xi_207
                vwtd_num_i_207 = wt_se_i_207*xi2_207
                wts207[i] = wt_se_i_207
                xbarwtd_num_207[i] = xbarwtd_num_i_207
                vwtd_num_207[i] = vwtd_num_i_207
                
            wtdmean = np.sum(xbarwtd_num)/np.sum(wts) # wtd mean
            wtdmean207 = np.sum(xbarwtd_num_207)/np.sum(wts207) # wtd mean for 7/35 ratio
            avg_reg_err = np.sqrt((np.sum(vwtd_num)/np.sum(wts) - (wtdmean)**2)*(len(df)/(len(df)-1))) # wtd error
            avg_reg_err_207 = np.sqrt((np.sum(vwtd_num_207)/np.sum(wts207) - (wtdmean207)**2)*(len(df)/(len(df)-1))) # wtd error for 7/35 ratio
        elif ellipse_mode_selector == 'Both':
            for i in range(0, len(df)):
                wt_se_i = 1/df['206/238 Reg. err'][i]
                xi = df['206Pb/238Upbc'][i]
                xi2 = df['206Pb/238Upbc'][i]**2
                xbarwtd_num_i = wt_se_i*xi
                vwtd_num_i = wt_se_i*xi2
                wts[i] = wt_se_i
                xbarwtd_num[i] = xbarwtd_num_i
                vwtd_num[i] = vwtd_num_i
                
                wt_se_i_207 = 1/df['207/235 Reg. err'][i]
                xi_207 = df['207Pb/235Upbc'][i]
                xi2_207 = df['207Pb/235Upbc'][i]**2
                xbarwtd_num_i_207 = wt_se_i_207*xi_207
                vwtd_num_i_207 = wt_se_i_207*xi2_207
                wts207[i] = wt_se_i_207
                xbarwtd_num_207[i] = xbarwtd_num_i_207
                vwtd_num_207[i] = vwtd_num_i_207
                
            wtdmean = np.sum(xbarwtd_num)/np.sum(wts)
            wtdmean207 = np.sum(xbarwtd_num_207)/np.sum(wts207)
            avg_reg_err = np.sqrt((np.sum(vwtd_num)/np.sum(wts) - (wtdmean)**2)*(len(df)/(len(df)-1)))
            avg_reg_err_207 = np.sqrt((np.sum(vwtd_num_207)/np.sum(wts207) - (wtdmean207)**2)*(len(df)/(len(df)-1)))
        else:
            avg_reg_err = np.mean(df['206/238 Reg. err'])
            avg_reg_err_207 = np.mean(df['207/235 Reg. err'])

        return avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UThstd, UTh_std_m, fracfactor_76
    
    
    
    def get_pt_ages(df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,
                    drift_treatment,drift_nearest):
        """
        Function that calculates ages of unknowns

        Parameters
        ----------
        df : pandas dataframe
            contains measured and reduced data of unkowns.
        std : pandas dataframe
            contains primary standard data.
        std_txt : string
            primary standard name.
        ThU_zrn : float
            input value for Th/U ratio in unknowns.
        ThU_magma : float
            input value for Th/U ratio in melt hosting unknowns.
        Pb_Th_std_crct_selector : string
            determines if correction should be for just common Pb or Common Pb and Th disequil.
        regression_selector : string
            string dneoting how Pb-U ratios were treated (1st order, exponential, total counts).
        ellipse_mode_selector : string
            string denoting to reduce point estimates, ellipsoids, or both.
        power : float
            alpha value for confidence ellipse.
        UTh_std_norm : string
            whether or not U/Th ratios have been calculated for each analysis. If so, use that ratio.
            Requires iterating through age calculation at least once.
        common_207206_input : float
            value for common Pb 7/6 ratio in common Pb correction.
        common_207206_error : float
            value for error on common Pb 7/6 ratio in common Pb correction.
        drift_treatment : string
            user requested option on how drift should be handled.
        drift_nearest : integer
            Nearest number of standard that should be used to correct data. i.e., sliding window of Gehrels et al. (2008)

        Returns
        -------
        df : pandas dataframe
            has all unknown data including reduced ages.
        """
        df = df.reset_index(drop=True) # reset indices
        zeros_like_df = np.zeros(len(df)) # set up array of zeros to populate pandas series that will be filled with calculations
        df['frac_factor_206238'] = zeros_like_df # initialize series
        df['frac_factor_207235'] = zeros_like_df
        df['tims_age_std'] = zeros_like_df
        df['tims_error_std'] = zeros_like_df
        df['tims_age_207'] = zeros_like_df
        df['tims_error_std_207'] = zeros_like_df
        df['avg_std_ratio'] = zeros_like_df
        df['avg_std_ratio_Thcrct'] = zeros_like_df
        df['avg_std_ratio_207'] = zeros_like_df
        df['avg_reg_err'] = zeros_like_df
        df['avg_reg_err_207'] = zeros_like_df
        # try to run the fucntion. allow user to keyboard interupt
        try:
            # if drift treatment requested, check if correcting by ZRM. If so, get the requested nearest number and use those to correct data
            if drift_treatment != 'None':
                if drift_treatment == 'By Age':
                    for i in range(0,len(df)):
                        nearest_stds = std.iloc[(std['measurementindex']-df.loc[i,'measurementindex']).abs().argsort()[:drift_nearest]] # get nearest standards
                        std_set_i = nearest_stds # variable change
                        # get the fractionation factors and standard statistics
                        frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m = calc_fncs.get_standard_fracfctr(std_set_i, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input)
                        df.loc[i,'frac_factor_206238'] = frac_factor # 6/38 fractionation factor
                        df.loc[i,'frac_factor_207235'] = frac_factor_207 # 7/35 fractionation factor
                        df.loc[i,'tims_age_std'] = tims_age # standard accepted age from TIMS
                        df.loc[i,'tims_error_std'] = tims_error # standard accepted age error from TIMS
                        df.loc[i,'tims_age_207'] = tims_age_207 # standard accepted 7/35 age age from TIMS
                        df.loc[i,'tims_error_std_207'] = tims_error_207 # standard accepted 7/35 age error from TIMS
                        df.loc[i,'avg_std_ratio'] = avg_std_ratio # average common Pb corrected 6/38 ratio from standard
                        df.loc[i,'avg_std_ratio_Thcrct'] = avg_std_ratio_Thcrct # average common Pb + Th. disequil. corrected 6/38 ratio from standard
                        df.loc[i,'avg_std_ratio_207'] = avg_std_ratio_207 # average common Pb corrected 7/35 ratio from standard
                        df.loc[i,'avg_reg_err'] = avg_reg_err # average 6/38 error on standard
                        df.loc[i,'avg_reg_err_207'] = avg_reg_err_207 # average 7/35 ratio from standard
                        
                else:
                    pass
                            
            else:
                frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m = calc_fncs.get_standard_fracfctr(std, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input)
                df['frac_factor_206238'] = frac_factor
                df['frac_factor_207235'] = frac_factor_207
                df['tims_age_std'] = tims_age
                df['tims_error_std'] = tims_error
                df['tims_age_207'] = tims_age_207
                df['tims_error_std_207'] = tims_error_207
                df['avg_std_ratio'] = avg_std_ratio
                df['avg_std_ratio_Thcrct'] = avg_std_ratio_Thcrct
                df['avg_std_ratio_207'] = avg_std_ratio_207
                df['avg_reg_err'] = avg_reg_err
                df['avg_reg_err_207'] = avg_reg_err_207
            
            
            df['238U/206Pb_corrected'] = 1/((1/df['238U/206Pb'])*df['frac_factor_206238']) # 38/6 bias corrected ratio
            df['207Pb/235U_corrected'] = df['207Pb/235U']*df['frac_factor_207235'] # 7/35 bias corrected artio
        except KeyboardInterrupt:
            print('Interrupted Age Calculations')
        
        # get points needed for correction
        points, concordia_238_206, pts_pb_r, n, a = calc_fncs.get_projections(df, ellipse_mode_selector, power, common_207206_input)
        
        # set up empty arrays to be filled for common lead corrections
        common_filter = np.zeros(len(df['SK 207Pb/206Pb']))
        
        pb_m = df['207Pb/206Pb c']  # measured or externally calculated mass bias corrected 207/206
        # if user input value for common Pb correction, assign that value, otherwise use default SK model
        if common_207206_input!= 0 and common_207206_error != 0:
            common = df['Common 207Pb/206Pb']
        else:
            common = df['SK 207Pb/206Pb']
        # if common Pb value not feasible, assign zero. Retain value otherwise
        for i in range(0,len(common)):
            if common[i] <= 0:
                common_filter[i] = 0
            else:
                common_filter[i] = common[i]

        # calculate fraction of common Pb
        f_ = (pb_m - pts_pb_r) / (common - pts_pb_r)
        # set up array to set f = 0 if point lies on or below Concordia (i.e., no common Pb present)
        f = np.zeros(len(f_))

        for k, j in zip(common_filter, range(0,len(f_))):
            if k <= 0:
                f[j] = 0
            elif f_[j] < 0:
                f[j] = 0
            else:
                f[j] = f_[j]

        # append the calculations to the sample dataframe
        df['207Pb/206Pbr'] = pts_pb_r
        df['f'] = f

        df['counts_pb206r'] = df['206Pb'] * (1-df['f']) # counts of radiogenic 206
        df['206Pb/238Upbc_numerical'] = 1 / df['238U/206Pb_corrected']-(1/df['238U/206Pb_corrected']*f) # numerically calculated 6/38 corrected ratio
        df['206Pb/238Upbc'] = 1/concordia_238_206 # projected 6/38 corrected ratio
        
        df['207Pb/235Upbc_corrected'] = df['207Pb/235U_corrected']-(df['207Pb/235U_corrected']*f) # numerically calculated 7/35 ratio
        df['207Pb/235Uc_age'] = np.log(df['207Pb/235Upbc_corrected'] + 1) / lambda_235 # 7/35 common Pb corrected age
        # check if user input values for common Pb correction. If so, propagate those errors. If not, use default value
        if common_207206_input!= 0 and common_207206_error != 0:
            dagetot_207 = np.abs(df['207Pb/235Uc_age'])*(((df['tims_error_std_207']/2)/df['tims_age_207'])**2 + (df['avg_reg_err_207']/df['avg_std_ratio_207'])**2 +
                                                             (df['207/235 Reg. err']/(df['207Pb/235U']))**2 + (lambda_235_2sig_percent/2/100)**2 +
                                                             (df['Common 207Pb/206Pb Error']/df['Common 207Pb/206Pb'])**2 +
                                                             (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
        else:
            dagetot_207 = np.abs(df['207Pb/235Uc_age'])*(((df['tims_error_std_207']/2)/df['tims_age_207'])**2 + (df['avg_reg_err_207']/df['avg_std_ratio_207'])**2 +
                                                             (df['207/235 Reg. err']/(df['207Pb/235U']))**2 + (lambda_235_2sig_percent/2/100)**2 +
                                                             ((SK74_2sig/2)/df['SK 207Pb/204Pb'])**2 + ((SK64_2sig/2)/df['SK 206Pb/204Pb'])**2 +
                                                             (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
        df['∆207/235 age (tot.)'] = dagetot_207
        # check which type of corrections user requests. propagate errors based on those
        if Pb_Th_std_crct_selector == 'Common Pb':
            df['206Pb/238U_correctedage'] = np.log(df['206Pb/238Upbc'] + 1) / lambda_238 # 6/38 common Pb corrected age
            # propagate errors
            # error on age equation. error on decay constant includes 1.5* counting stats (Mattinson 1987)
            # error on estimation for common lead using 207 method. Uses conservaitve estimates of 1.0 for 206/204 and 0.3 for 207/204 (Mattionson, 1987)
            # total propagated error
            dage = np.abs(df['206Pb/238U_correctedage'])*((df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2) # analytical error
            # fully propagated errors
            if common_207206_input!= 0 and common_207206_error != 0:
                dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                 (df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                 (df['Common 207Pb/206Pb Error']/df['Common 207Pb/206Pb'])**2 +
                                                                 (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
            else:
                dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                 (df['206/238 Reg. err']/(df['206Pb/238U_unc']))**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                 ((SK74_2sig/2)/df['SK 207Pb/204Pb'])**2 + ((SK64_2sig/2)/df['SK 206Pb/204Pb'])**2 +
                                                                 (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)

            df['∆206/238 age (meas.)'] = dage
            df['∆206/238 age (tot.)'] = dagetot
        elif Pb_Th_std_crct_selector == 'Common Pb + Th Disequil.':
            if UTh_std_norm == 'Off':
                df['206Pb/238UPbThc'] = df['206Pb/238Upbc'] - (lambda_238/lambda_230*((ThU_zrn/ThU_magma)-1)) # common Pb and Th disequil corrected 6/38 ratio
                df['206Pb/238U_correctedage'] = (np.log(df['206Pb/238UPbThc'] + 1) / lambda_238) # common Pb and Th disequil corrected 6/38 age
                # propagate errors
                # error on fractionation factor. includes error from ID-TIMS and ICPMS
                # error on age equation. error on decay constant includes 1.5* counting stats (Mattinson 1987)
                # error on estimation for common lead using 207 method. Uses conservaitve estimates of 1.0 for 206/204 and 0.3 for 207/204 (Mattionson, 1987)
                # UTh errors - only using errors from measured zircon here, as this will by and large be the largest error contribution
                # should probably add error for U / Th in icpms glass analyses, though rock measurements will undoubtedly be incorrectly used by users of the
                # program (in absence of other data) and so added error would overall be fairly misleading anyway
                # errors on absolute concentrations from ID-TIMS are overall negligible and often not reported either.
                # need to put in possibility of putting in errors on Th/U measurements
                # total propagated error
                dage = np.abs(df['206Pb/238U_correctedage'])*((df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2) # analytical error
                # fully propagated errors
                if common_207206_input!= 0 and common_207206_error != 0:
                    dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                     (df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                     (df['Common 207Pb/206Pb Error']/df['Common 207Pb/206Pb'])**2 +
                                                                     (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
                else:
                    dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                     (df['206/238 Reg. err']/(df['206Pb/238U_unc']))**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                     ((SK74_2sig/2)/df['SK 207Pb/204Pb'])**2 + ((SK64_2sig/2)/df['SK 206Pb/204Pb'])**2 +
                                                                     (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)

                df['∆206/238 age (meas.)'] = dage
                df['∆206/238 age (tot.)'] = dagetot
            elif UTh_std_norm == 'Calc U/Th':
                ThUzrn_calc = 1/df['238U/232Th_calc']
                df['206Pb/238UPbThc'] = df['206Pb/238Upbc'] - (lambda_238/lambda_230*((ThUzrn_calc/ThU_magma)-1))
                df['206Pb/238U_correctedage'] = (np.log(df['206Pb/238UPbThc'] + 1) / lambda_238)
                # propagate errors
                # error on fractionation factor. includes error from ID-TIMS and ICPMS
                # error on age equation. error on decay constant includes 1.5* counting stats (Mattinson 1987)
                # error on estimation for common lead using 207 method. Uses conservaitve estimates of 1.0 for 206/204 and 0.3 for 207/204 (Mattionson, 1987)
                # UTh errors
                # total propagated error
                dage = np.abs(df['206Pb/238U_correctedage'])*((df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
                if common_207206_input!= 0 and common_207206_error != 0:
                    dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                     (df['206/238 Reg. err']/df['206Pb/238U_unc'])**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                     (df['Common 207Pb/206Pb Error']/df['Common 207Pb/206Pb'])**2 +
                                                                     (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
                else:
                    dagetot = np.abs(df['206Pb/238U_correctedage'])*(((df['tims_error_std']/2)/df['tims_age_std'])**2 + (df['avg_reg_err']/df['avg_std_ratio_Thcrct'])**2 +
                                                                     (df['206/238 Reg. err']/(df['206Pb/238U_unc']))**2 + (lambda_238_2sig_percent/2/100)**2 +
                                                                     ((SK74_2sig/2)/df['SK 207Pb/204Pb'])**2 + ((SK64_2sig/2)/df['SK 206Pb/204Pb'])**2 +
                                                                     (df['SE 207/206']/df['207Pb/206Pb'])**2)**(1/2)
                df['∆206/238 age (meas.)'] = dage
                df['∆206/238 age (tot.)'] = dagetot

        return df
        
        

    def get_ellipse_ages(ell_df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,
                         drift_treatment,drift_nearest):
        """
        Function that corrects ellipsoids for mass bias similar to those for point estimates

        Parameters
        ----------
        ell_df : pandas dataframe
            df that holds the integrations.
        std : pandas dataframe
            df that holds primary standard values. 
        std_txt : string
            primary standard name.
        ThU_zrn : float
            input value for Th/U ratio in unknowns.
        ThU_magma : float
            input value for Th/U ratio in melt hosting unknowns.
        Pb_Th_std_crct_selector : string
            determines if correction should be for just common Pb or Common Pb and Th disequil.
        regression_selector : string
            string dneoting how Pb-U ratios were treated (1st order, exponential, total counts).
        ellipse_mode_selector : string
            string denoting to reduce point estimates, ellipsoids, or both.
        power : float
            alpha value for confidence ellipse.
        UTh_std_norm : string
            whether or not U/Th ratios have been calculated for each analysis. If so, use that ratio.
            Requires iterating through age calculation at least once.
        common_207206_input : float
            value for common Pb 7/6 ratio in common Pb correction.
        common_207206_error : float
            value for error on common Pb 7/6 ratio in common Pb correction.
        drift_treatment : string
            user requested option on how drift should be handled.
        drift_nearest : integer
            Nearest number of standard that should be used to correct data. i.e., sliding window of Gehrels et al. (2008)

        Returns
        -------
        ellparams_i : pandas dataframe
            dataframe with corrected ellipsoid parameters to be projected onto Tera-Wasserburg Concordia.
        ellparams_iw : pandas dataframe
            dataframe with corrected ellipsoid parameters to be projected onto TWetherhill Concordia..

        """
        ell_df = ell_df.reset_index(drop=True)
        zeros_like_df = np.zeros(len(ell_df))
        ell_df['frac_factor_206238'] = zeros_like_df
        ell_df['frac_factor_207235'] = zeros_like_df
        ell_df['frac_factor_76'] = zeros_like_df
        ell_df['tims_age_std'] = zeros_like_df
        ell_df['tims_error_std'] = zeros_like_df
        ell_df['tims_age_207'] = zeros_like_df
        ell_df['tims_error_std_207'] = zeros_like_df
        ell_df['avg_std_ratio'] = zeros_like_df
        ell_df['avg_std_ratio_Thcrct'] = zeros_like_df
        ell_df['avg_std_ratio_207'] = zeros_like_df
        ell_df['avg_reg_err'] = zeros_like_df
        ell_df['avg_reg_err_207'] = zeros_like_df
        try:
            if drift_treatment != 'None':
                if drift_treatment == 'By Age':
                    for i in ell_df.SampleLabel.unique():
                        unique_analysis = ell_df[ell_df['SampleLabel'] == i]
                        minindex = unique_analysis.index[0]
                        nearest_stds = std.iloc[(std['measurementindex']-unique_analysis.loc[minindex,'measurementindex']).abs().argsort()[:drift_nearest]]
                        std_set_i = nearest_stds
                        frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m = calc_fncs.get_standard_fracfctr(std_set_i, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input)
                        unique_analysis.loc[:,'frac_factor_206238'] = frac_factor
                        unique_analysis.loc[:,'frac_factor_207235'] = frac_factor_207
                        unique_analysis.loc[:,'frac_factor_76'] = fracfactor_76
                        unique_analysis.loc[:,'tims_age_std'] = tims_age
                        unique_analysis.loc[:,'tims_error_std'] = tims_error
                        unique_analysis.loc[:,'tims_age_207'] = tims_age_207
                        unique_analysis.loc[:,'tims_error_std_207'] = tims_error_207
                        unique_analysis.loc[:,'avg_std_ratio'] = avg_std_ratio
                        unique_analysis.loc[:,'avg_std_ratio_Thcrct'] = avg_std_ratio_Thcrct
                        unique_analysis.loc[:,'avg_std_ratio_207'] = avg_std_ratio_207
                        unique_analysis.loc[:,'avg_reg_err'] = avg_reg_err
                        unique_analysis.loc[:,'avg_reg_err_207'] = avg_reg_err_207

                        ell_df.loc[unique_analysis.index,:] = ell_df.loc[unique_analysis.index,:].mask(ell_df.loc[unique_analysis.index,:]!=unique_analysis.loc[unique_analysis.index,:],unique_analysis.loc[unique_analysis.index,:])
                else:
                    pass
                            
            else:
                frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m = calc_fncs.get_standard_fracfctr(std, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input)
                ell_df['frac_factor_206238'] = frac_factor
                ell_df['frac_factor_207235'] = frac_factor_207
                ell_df['frac_factor_76'] = fracfactor_76
                ell_df['tims_age_std'] = tims_age
                ell_df['tims_error_std'] = tims_error
                ell_df['tims_age_207'] = tims_age_207
                ell_df['tims_error_std_207'] = tims_error_207
                ell_df['avg_std_ratio'] = avg_std_ratio
                ell_df['avg_std_ratio_Thcrct'] = avg_std_ratio_Thcrct
                ell_df['avg_std_ratio_207'] = avg_std_ratio_207
                ell_df['avg_reg_err'] = avg_reg_err
                ell_df['avg_reg_err_207'] = avg_reg_err_207
            
            
            ell_df['238U/206Pb_corrected'] = 1/((1/ell_df['238U/206Pb'])*ell_df['frac_factor_206238'])
            ell_df['207Pb/235U_corrected'] = ell_df['207Pb/235U']*ell_df['frac_factor_207235']
        except KeyboardInterrupt:
            print('Interrupted Age Calculations')
            
            
        
        frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m \
            = calc_fncs.get_standard_fracfctr(std, std_txt, Pb_Th_std_crct_selector, regression_selector,ellipse_mode_selector, power, common_207206_input)

        # set up empty arrays to be filled for common lead corrections
        
        # if an external on ZRM such as nist was not used to correct 7/6 bias, correct by the concordant ZRM 7/6 ratio. Needs to overwrite raw value
        # otherwise pass as this would have been calculated already
        if pb_bias_type == 'By Age':
            pb_m = ell_df['207Pb/206Pb c']
            ell_df['207Pb/206Pb c'] = ell_df['207Pb/206Pb c']*(mass_dict.get('207Pb')/mass_dict.get('206Pb'))**fracfactor_76
        else:
            pass
        
        ell_df['206Pb/238U_corrected'] = 1/ell_df['238U/206Pb_corrected']

        if Pb_Th_std_crct_selector == 'Common Pb':
            ell_df['206Pb/238U_correctedage'] = np.log(ell_df['206Pb/238U_corrected'] + 1) / lambda_238

            ellparams_i,ellparams_iw = calc_fncs.get_ellipse(ell_df, power)
            ellparams_i = pd.DataFrame([ellparams_i], columns=['Ellipse Center', 'Ell. Width', 'Ell. Height', 'Ell. Rotation'])
            ellparams_iw = pd.DataFrame([ellparams_iw], columns=['Ellipse Center Weth', 'Ell. Width Weth', 'Ell. Height Weth', 'Ell. Rotation Weth'])

        elif Pb_Th_std_crct_selector == 'Common Pb + Th Disequil.':
            if UTh_std_norm == 'Off':
                ell_df['206Pb/238UPbThc'] = ell_df['206Pb/238U_corrected'] - (lambda_238/lambda_230*((ThU_zrn/ThU_magma)-1))
                ell_df['206Pb/238U_correctedage'] = (np.log(ell_df['206Pb/238UPbThc'] + 1) / lambda_238)

                ellparams_i,ellparams_iw = calc_fncs.get_ellipse(ell_df, power)
                ellparams_i = pd.DataFrame([ellparams_i], columns=['Ellipse Center', 'Ell. Width', 'Ell. Height', 'Ell. Rotation'])
                ellparams_iw = pd.DataFrame([ellparams_iw], columns=['Ellipse Center Weth', 'Ell. Width Weth', 'Ell. Height Weth', 'Ell. Rotation Weth'])

            elif UTh_std_norm == 'Calc U/Th':
                ThUzrn_calc = 1/ell_df['238U/232Th_calc']
                ell_df['206Pb/238UPbThc'] = ell_df['206Pb/238U_corrected'] - (lambda_238/lambda_230*((ThUzrn_calc/ThU_magma)-1))
                ell_df['206Pb/238Uc_age'] = (np.log(ell_df['206Pb/238UPbThc'] + 1) / lambda_238)
                ell_df['206Pb/238U_correctedage'] = ell_df['206Pb/238Uc_age']*frac_factor
                
                ellparams_i,ellparams_iw = calc_fncs.get_ellipse(ell_df, power)
                ellparams_i = pd.DataFrame([ellparams_i], columns=['Ellipse Center', 'Ell. Width', 'Ell. Height', 'Ell. Rotation'])
                ellparams_iw = pd.DataFrame([ellparams_iw], columns=['Ellipse Center Weth', 'Ell. Width Weth', 'Ell. Height Weth', 'Ell. Rotation Weth'])
        return ellparams_i,ellparams_iw
        
        

    def correct_sample_ages(df, ell_df, std, std_ell, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,drift_treatment,drift_nearest):
        """
        Function used to correct send unknown samples to the methods that get the fully corrected ages. 
        Essentially a funnel for handling the different data types in the unkown points estimates, unknown ellipsoids, and standard points estimates / ellipsoids

        Parameters
        ----------
        df : pandas dataframe
            df that holds point estimates of unknowns.
        ell_df : pandas dataframe
            df that holds the integrations.
        std : pandas dataframe
            df that holds primary standard values. 
        std_ell : pandas dataframe
            df that holds ellipsoid data for standards.
        std_txt : string
            primary standard name.
        ThU_zrn : float
            input value for Th/U ratio in unknowns.
        ThU_magma : float
            input value for Th/U ratio in melt hosting unknowns.
        Pb_Th_std_crct_selector : string
            determines if correction should be for just common Pb or Common Pb and Th disequil.
        regression_selector : string
            string dneoting how Pb-U ratios were treated (1st order, exponential, total counts).
        ellipse_mode_selector : string
            string denoting to reduce point estimates, ellipsoids, or both.
        power : float
            alpha value for confidence ellipse.
        UTh_std_norm : string
            whether or not U/Th ratios have been calculated for each analysis. If so, use that ratio.
            Requires iterating through age calculation at least once.
        common_207206_input : float
            value for common Pb 7/6 ratio in common Pb correction.
        common_207206_error : float
            value for error on common Pb 7/6 ratio in common Pb correction.
        drift_treatment : string
            user requested option on how drift should be handled.
        drift_nearest : integer
            Nearest number of standard that should be used to correct data. i.e., sliding window of Gehrels et al. (2008)
            
        Returns
        -------
        dataframes
            Depends on ellipse_mode_selector. Returns  reduced data for point estimates, ellipsoids, or both

        """
        if ellipse_mode_selector == 'Point Estimates':
            pt_ages = calc_fncs.get_pt_ages(df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,drift_treatment,drift_nearest)
            
            # print('Got Point Ages')
            return pt_ages
        elif ellipse_mode_selector == 'Ellipses':
            ellparams_i,ellparams_iw = calc_fncs.get_ellipse_ages(ell_df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,drift_treatment,drift_nearest)
            
            return ellparams_i,ellparams_iw
        
        elif ellipse_mode_selector == 'Both':
            pt_ages = calc_fncs.get_pt_ages(df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,drift_treatment,drift_nearest)
            ellparams_i,ellparams_iw = calc_fncs.get_ellipse_ages(ell_df, std, std_txt, ThU_zrn, ThU_magma, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, UTh_std_norm, common_207206_input,common_207206_error,drift_treatment,drift_nearest)
            # print('Got Everything')

            return pt_ages,ellparams_i,ellparams_iw

            

    def get_standard_fracfctr(std, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input):
        """
        function that gets fractionantion factors and standard statistics or accepted values
        first set of accepted ages and errors are 206/238 ages. second are 207/235 ages

        Parameters
        ----------
        std : pandas dataframe
            df that holds standard data.
        std_txt : string
            has standard name.
        Pb_Th_std_crct_selector : string
            allows user to denote Common Pb correction or Common Pb and Th disequil.
        regression_selector : string
            user selection on how Pb-U ratios were treated.
        ellipse_mode_selector : string
            user selection on whether to use point estimates, ellipsoids, or both.
        power : float
            alpha value on confidence ellipsoids.
        common_207206_input : float
            common Pb correction ratio. Used for testing

        Returns
        -------
        frac_factor : float
            fractionation factor on 6/38 age.
        frac_factor_207 : float
            fractionation factor on 7/35 age.
        fracfactor_76 : float
            fractionation factor on 7/6 ratio.
        tims_age : float
            accepted age from tims.
        tims_error : float
            accepted error from tims.
        tims_age_207 : float
            accepted 7/35 age from tims.
        tims_error_207 : float
            accepted 7/35 error from tims.
        avg_std_age : float
            average age of standard.
        avg_std_age_Thcrct : float
            average age of standard with Th correction.
        avg_std_age_207 : float
            average 207/235 age of standard.
        avg_std_ratio : float
            average 206/238 ratio of standard.
        avg_std_ratio_Thcrct : float
            average 206/238 ratio of standard with Th correction.
        avg_std_ratio_207 : float
            average 207/235 ratio of standard.
        avg_reg_err : float
            average error on the 206/238 ratio.
        avg_reg_err_207 : float
            average error on the 207/235 ratio.
        UTh_std : float
            accepted U/Th ratio in standards
        UTh_std_m : float
            measured U/Th ratio in standards.

        """
        # correct standard ages, get frac factors, etc
        avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m, fracfactor_76 = \
            calc_fncs.correct_standard_ages(std, std_txt, Pb_Th_std_crct_selector, regression_selector, ellipse_mode_selector, power, common_207206_input)
        tims_age = accepted_ages.get(std_txt)[0] # get accepted from dictionary
        tims_error = TIMS_errors.get(std_txt)[0]
        tims_age_207 = accepted_ages.get(std_txt)[1]
        tims_error_207 = TIMS_errors.get(std_txt)[1]
        std_accpt_ratio_207 = np.exp(tims_age_207*lambda_235)-1 # calculate accepted ratio
        frac_factor_207 = std_accpt_ratio_207/avg_std_ratio_207 # get frac factor. Very rudimentary at this stage
        std_accpt_ratio = np.exp(tims_age*lambda_238)-1 # calculate accepted ratio from TIMS accepted age

        if Pb_Th_std_crct_selector == 'Common Pb':
            frac_factor = std_accpt_ratio/avg_std_ratio
        elif Pb_Th_std_crct_selector == 'Common Pb + Th Disequil.':
            frac_factor = std_accpt_ratio/avg_std_ratio_Thcrct

        return frac_factor, frac_factor_207, fracfactor_76, tims_age, tims_error, tims_age_207, tims_error_207, avg_std_age, avg_std_age_Thcrct, avg_std_age_207, avg_std_ratio, avg_std_ratio_Thcrct, avg_std_ratio_207, avg_reg_err, avg_reg_err_207, UTh_std, UTh_std_m


# %%
class finalize_ages(param.Parameterized):

    """ class that parameterizes inputs and sends them to the above functions to be rendered in a GUI"""
    file_path = param.String(default='Insert File Path')
    input_data = param.DataFrame(precedence=-1)
    output_data = param.DataFrame(precedence=-1)
    output_secondary_data = param.DataFrame(precedence=-1)
    # drift_data = param.DataFrame(precedence=-1)
    # file_path_ellipse = param.String(default='Insert Ellipse File Path')
    input_data_ellipse = param.DataFrame(precedence=-1)
    output_data_ellipse = param.DataFrame(precedence=-1)
    regression_selector = param.Selector(objects=['1st Order', 'Exp', 'Total Counts'],precedence=-1)
    ellipse_mode_selector = param.Selector(default='Point Estimates', objects=['Point Estimates', 'Ellipses','Both'],precedence=-1)
    # need to output secondary figs automatically to assess data quality easily..
    # need to allow plot color changes
    # need to get ellipse data completely overhauled
    # need to make the export plot buttons work
    drift_analyte_dropdown = param.Selector(default='206Pb/238U',objects=['206Pb/238U Age','207Pb/235U Age',
                                                                          '206Pb/238U','207Pb/235U','238U/235U',
                                                                          '207Pb/206Pb: By Measured Mass Bias','207Pb/206Pb: By Age'])
    
    x_axis_TW = param.Range(default=(0,25),bounds=(0,10000))
    y_axis_TW = param.Range(default=(0,0.3),bounds=(0,1))
    
    x_axis_Weth= param.Range(default=(0,40),bounds=(0,1000))
    y_axis_Weth = param.Range(default=(0,0.8),bounds=(0,10))

    text_sample_selector = param.String(default='Input Sample ID')
    text_standard_selector = param.String(default='Input Standard ID',precedence=-1)
    secondary_std_selector = param.ListSelector(default=[],objects=[],precedence=-1)
    drift_selector = param.Selector(default='By Age',objects=['By Age','By Mass Bias and Age','None'],precedence=-1)
    drift_nearest_amount = param.Number(4,precedence=-1)
    common_207206_input = param.Number()
    common_207206_error = param.Number()
    ThU_zrn_input = param.Number()
    ThU_magma_input = param.Number()
    UTh_std_norm = param.Selector(default='Off', objects=['Calc U/Th', 'Off'])
    Pb_Th_std_crct_selector = param.Selector(objects=['Common Pb', 'Common Pb + Th Disequil.'])
    mass_bias_pb = param.Selector(objects=['By Age','NIST-614','NIST-612','NIST-610'],precedence=-1)
    mass_bias_pb_ratio = param.Selector(objects=['206Pb/204Pb','207Pb/204Pb','208Pb/204Pb','207Pb/206Pb'],precedence=-1)
    mass_bias_thu = param.Selector(objects=['Primary 238U/235U','NIST-614','NIST-612','NIST-610','None'],precedence=-1)
    mass_bias_thu_ratio = param.Selector(objects=['238U/235U'],precedence=-1)
    power = param.Number(default=0.05)

    update_output_button = param.Action(lambda x: x.add_output_data(), label='Approve Data')
    export_data_button = param.Action(lambda x: x.export_data(), label='DDDT!')
    accept_reduction_parameters_button = param.Action(lambda x: x._accept_reduction_parameters(),label='Accept Reduction Parameters')
    export_TWplot_button = param.Action(lambda x: x.export_plot(), label='Save Plot')
    label_toggle = param.ListSelector(default=['Concordia'], objects=['Concordia', 'Box + Whisker'])

    def __init__(self, **params):
        super().__init__(**params)
        self.input_data_widget = pn.Param(self.param.input_data),
        self.output_data_widget = pn.Param(self.param.output_data),
        self.output_secondary_data_widget = pn.Param(self.param.output_secondary_data),
        self.input_data_ellipse_widget = pn.Param(
            self.param.input_data_ellipse)
        self.output_data_ellipse_widget = pn.Param(
            self.param.output_data_ellipse)
        self.widgets = pn.Param(self, parameters=['label_toggle', 'regression_selector', 'ellipse_mode_selector','drift_analyte_dropdown','secondary_std_selector',
                                                  'update_output_button', 'export_data_button', 'export_TWplot_button',
                                                  'x_axis_TW','y_axis_TW','x_axis_Weth','y_axis_Weth',
                                                  'common_207206_input', 'common_207206_error',
                                                  'ThU_zrn_input', 'ThU_magma_input', 'power'])

    @pn.depends('file_path',watch=True)
    def _uploadfile(self):
        if self.file_path != 'Insert File Path':
            df = pd.read_excel(self.file_path, sheet_name='Sheet1')
            self.input_data = df
            self.input_data = self.input_data.replace('bdl',0)
            self.input_data[['Sample','Sample Analysis Number']] = df['SampleLabel'].str.rsplit('-',n=1,expand=True)
            unique_labels = self.input_data['Sample'].unique().tolist()
            fastgrid_layout.modal[0].append(pn.Column(pn.widgets.AutocompleteInput(name='Primary Standard',options=unique_labels,
                                                                                   case_sensitive=False))
                                            )
            fastgrid_layout.modal[1].append(pn.Row(pn.widgets.CheckBoxGroup(name='Secondary Standards',options=unique_labels,inline=True))
                                            )
            fastgrid_layout.modal[2].append(pn.Column(pn.Row(pn.widgets.RadioButtonGroup(name='Regression Selector',options=['1st Order', 'Exp', 'Total Counts'])),
                                                      pn.Row(pn.widgets.RadioButtonGroup(name='Data Type',options=['Point Estimates', 'Ellipses', 'Both'])),
                                                      pn.Row(pn.widgets.TextInput(name='Ellipse File Input',placeholder='Input Ellipse File'))
                                                      )
                                            )
            fastgrid_layout.modal[3].append(pn.Column(pn.Row(pn.widgets.RadioButtonGroup(name='Mass Bias Pb',options=['By Age','NIST-614','NIST-612','NIST-610'])),
                                                      pn.Row(pn.widgets.RadioButtonGroup(name='Mass Bias Pb Ratio',options=['206Pb/204Pb','207Pb/204Pb','208Pb/204Pb','207Pb/206Pb'])),
                                                      pn.Row(pn.widgets.RadioButtonGroup(name='Mass Bias U & Th',options=['Primary 238U/235U','NIST-614','NIST-612','NIST-610','None'])),
                                                      pn.Row(pn.widgets.RadioButtonGroup(name='Mass Bias U & Th Ratio',options=['238U/235U']))
                                                      )
                                            )
            fastgrid_layout.modal[4].append(pn.Column(pn.Row(pn.widgets.RadioButtonGroup(name='Drift Type',options=['By Age','None'])),
                                                      pn.Row(pn.widgets.IntInput(name='Nearest Number',value=4,step=1,start=1,end=1000))
                                                      )
                                            )
            fastgrid_layout.modal[5].append(pn.Column(pn.Row(pn.widgets.RadioButtonGroup(name='Common Pb and U-Th Disequil. Correction',options=['Common Pb', 'Common Pb + Th Disequil.'])),
                                                      pn.Row(pn.widgets.RadioButtonGroup(name='Calc U/Th from Primary?',options=['Calc U/Th', 'Off'])),
                                                      pn.Row(pn.widgets.FloatInput(name='Th/U Zircon',value=1)),
                                                      pn.Row(pn.widgets.FloatInput(name='Th/U Magma',value=3.03))
                                                      )
                                            )
            fastgrid_layout.modal[6].append(pn.Row(modal_button_one))
            fastgrid_layout.open_modal()
                
    @pn.depends('regression_selector', 'ellipse_mode_selector', 'drift_selector','drift_nearest_amount',
                'text_standard_selector','secondary_std_selector',
                'mass_bias_pb','mass_bias_pb_ratio','mass_bias_thu','mass_bias_thu_ratio',
                'Pb_Th_std_crct_selector','ThU_zrn_input','ThU_magma_input')
    def _accept_reduction_parameters(self,event=None):
        u_pb_ratio_treatment = fastgrid_layout.modal[2][0][0][0].value
        data_reduction_treatment = fastgrid_layout.modal[2][0][1][0].value
        ellipse_file = fastgrid_layout.modal[2][0][2][0].value
        global pb_bias_type
        pb_bias_type = fastgrid_layout.modal[3][0][0][0].value
        pb_bias_ratio = fastgrid_layout.modal[3][0][1][0].value
        u_bias_type = fastgrid_layout.modal[3][0][2][0].value
        u_bias_ratio = fastgrid_layout.modal[3][0][3][0].value
        primary_std = fastgrid_layout.modal[0][0][0].value
        drift_treatment = fastgrid_layout.modal[4][0][0][0].value
        drift_nearest = fastgrid_layout.modal[4][0][1][0].value
        secondary_standard_list = fastgrid_layout.modal[1][0][0].value
        commonPb_Thdisequil_treatment_stds = fastgrid_layout.modal[5][0][0][0].value
        UTh_disequil_stds = fastgrid_layout.modal[5][0][1][0].value
        ThUzirconratio_stds = fastgrid_layout.modal[5][0][2][0].value
        ThUmagmaratio_stds = fastgrid_layout.modal[5][0][3][0].value
        

        self.text_standard_selector = primary_std
        self.secondary_std_selector = secondary_standard_list
        self.regression_selector = u_pb_ratio_treatment
        self.ellipse_mode_selector = data_reduction_treatment
        self.drift_selector = drift_treatment
        self.drift_nearest_amount = drift_nearest
        self.mass_bias_pb = pb_bias_type
        self.mass_bias_pb_ratio = pb_bias_ratio
        self.u_bias_type = u_bias_type
        self.mass_bias_thu_ratio = u_bias_ratio
        
        
        
        
        if (data_reduction_treatment == 'Point Estimates' or data_reduction_treatment == 'Both'):
            if '206/238 1st Order' in self.input_data.columns and u_pb_ratio_treatment == '1st Order':
                self.input_data['238U/206Pb'] = 1 / self.input_data['206/238 1st Order']
                self.input_data['206Pb/238U_unc'] = self.input_data['206/238 1st Order']
                self.input_data['206/238 Reg. err'] = self.input_data['SE 206/238 1st Order']
                self.input_data['238/206 err'] = self.input_data['SE 206/238 1st Order']
                self.input_data['207Pb/235U'] = self.input_data['207/235 1st Order']
                self.input_data['207/235 Reg. err'] = self.input_data['SE 207/235 1st Order']
                print('1st Order Regression Selected')
            elif '206/238 Exp.' in self.input_data.columns and u_pb_ratio_treatment == 'Exp':
                self.input_data['238U/206Pb'] = 1 / self.input_data['206/238 Exp.']
                self.input_data['206Pb/238U_unc'] = self.input_data['206/238 Exp.']
                self.input_data['206/238 Reg. err'] = self.input_data['SE 206/238 Exp']
                self.input_data['238/206 err'] = self.input_data['SE 206/238 Exp']
                self.input_data['207Pb/235U'] = self.input_data['207/235 Exp.']
                self.input_data['207/235 Reg. err'] = self.input_data['SE 207/235 Exp']
                print('Exponential Regression Selected')
            elif '206Pb/238U' in self.input_data.columns and u_pb_ratio_treatment == 'Total Counts':
                self.input_data['238U/206Pb'] = 1 / self.input_data['206Pb/238U']
                self.input_data['206Pb/238U_unc'] = self.input_data['206Pb/238U']
                self.input_data['206/238 Reg. err'] = self.input_data['SE 206/238']
                self.input_data['238/206 err'] = self.input_data['SE 238U/206Pb']
                self.input_data['207Pb/235U'] = self.input_data['207Pb/235U']
                self.input_data['207/235 Reg. err'] = self.input_data['SE 207/235']
                print('Total Counts Selected')
            else:
                pass
            
            self.input_data['206/238U_age_init'] = np.log((1/self.input_data['238U/206Pb']) + 1) / lambda_238
            self.input_data['207/235U_age_init'] = np.log(self.input_data['207Pb/235U'] + 1) / lambda_235
            self.output_data = pd.DataFrame([np.zeros(len(self.input_data.columns))], columns=list(self.input_data.columns))
            
            if ellipse_file != '':
                pass_ell_df = pd.read_excel(ellipse_file,sheet_name='Sheet1')
                self.input_data_ellipse = pass_ell_df
                self.input_data_ellipse[['Sample','Sample Analysis Number']] = self.input_data_ellipse['SampleLabel'].str.rsplit('-',n=1,expand=True)
                col_bdl_condn = self.input_data_ellipse[(self.input_data_ellipse['206Pb/238U'] == 'bdl') | (self.input_data_ellipse['207Pb/206Pb'] == 'bdl') | (self.input_data_ellipse['207Pb/235U'] == 'bdl') | (self.input_data_ellipse['238U/235U'] == 'bdl')].index
                self.input_data_ellipse = self.input_data_ellipse.drop(col_bdl_condn, inplace=False)
                self.input_data_ellipse = self.input_data_ellipse.reset_index(drop=True)
                self.input_data_ellipse['206Pb/238U'] = pd.to_numeric(self.input_data_ellipse['206Pb/238U'])
                self.input_data_ellipse['207Pb/206Pb'] = pd.to_numeric(self.input_data_ellipse['207Pb/206Pb'])
                self.input_data_ellipse['207Pb/235U'] = pd.to_numeric(self.input_data_ellipse['207Pb/235U'])
                self.input_data_ellipse['238U/206Pb'] = 1 / self.input_data_ellipse['206Pb/238U']
                self.input_data_ellipse['206/238U_age_init'] = np.log(self.input_data_ellipse['206Pb/238U'] + 1) / lambda_238
                if '238U/232Th_calc' in self.input_data.columns:
                    self.input_data_ellipse = self.input_data_ellipse.merge(self.input_data[['SampleLabel','238U/232Th_calc']],on='SampleLabel',how='left')
                print('READ ELLIPSE FILE CORRECTLY')
                self.output_data_ellipse = pd.DataFrame([np.zeros(10)], columns=['measurementindex','SampleLabel',
                                                                                'Ellipse Center', 'Ell. Width', 'Ell. Height', 'Ell. Rotation',
                                                                                'Ellipse Center Weth', 'Ell. Width Weth', 'Ell. Height Weth', 'Ell. Rotation Weth'])
                
            if pb_bias_type != 'By Age':
                pb_bias_std = pb_bias_type
                pb_bias_std_df = self.input_data[self.input_data['Sample'] == pb_bias_std]
                pb_bias_std_df = pb_bias_std_df.reset_index(drop=True)
                high_mass_pb,low_mass_pb = pb_bias_ratio.split('/')
                high_mass_pb_wt = mass_dict.get(high_mass_pb)
                low_mass_pb_wt = mass_dict.get(low_mass_pb)
                accepted_pb = pb_bias_dict[pb_bias_std][pb_bias_ratio]
                pb_f = np.log(accepted_pb/np.mean(pb_bias_std_df[pb_bias_ratio]))/np.log(high_mass_pb_wt/low_mass_pb_wt)
                if high_mass_pb == '207Pb' and low_mass_pb == '206Pb':
                    self.input_data['207Pb/206Pb c'] = self.input_data['207Pb/206Pb']*(high_mass_pb_wt/low_mass_pb_wt)**pb_f
                    accepted_pb_64 = pb_bias_dict[pb_bias_std]['206Pb/204Pb']
                    m206 = mass_dict.get('206Pb')
                    m204 = mass_dict.get('204Pb')
                    pb_f_64 = pb_f = np.log(accepted_pb_64/np.mean(pb_bias_std_df['206Pb/204Pb']))/np.log(m206/m204)
                    self.input_data['206Pb/204Pb c'] = self.input_data['206Pb/204Pb']*(m206/m204)**pb_f_64
                    if self.ellipse_mode_selector != 'Point Estimates':
                        self.input_data_ellipse['207Pb/206Pb c'] = self.input_data_ellipse['207Pb/206Pb']*(high_mass_pb_wt/low_mass_pb_wt)**pb_f
                        self.input_data_ellipse['207Pb/204Pb c'] = self.input_data_ellipse['206Pb/204Pb']*(m206/m204)**pb_f_64
                print('Pb Bias Calculated Externally')
            else:
                self.input_data['207Pb/206Pb c'] = self.input_data['207Pb/206Pb']
                if self.ellipse_mode_selector != 'Point Estimates':
                    self.input_data_ellipse['207Pb/206Pb c'] = self.input_data_ellipse['207Pb/206Pb']
                print('Pb Bias Calculated by Ages')

                
            if u_bias_type != 'None':
                high_mass_u = '238U'
                low_mass_u = '235U'
                high_mass_u_wt = mass_dict.get(high_mass_u)
                low_mass_u_wt = mass_dict.get(low_mass_u)
                if u_bias_type == 'Primary 238U/235U':
                    u_bias_std = primary_std
                    u_bias_std_df = self.input_data[self.input_data['Sample'] == u_bias_std]
                    u_bias_std_df = u_bias_std_df.reset_index(drop=True)
                    accepted_u = 137.818
                    u_f = np.log(accepted_u/np.mean(u_bias_std_df['238U/235U']))/np.log(high_mass_u_wt/low_mass_u_wt)
                    self.input_data['238U/235U c'] = self.input_data['238U/235U']*(high_mass_u_wt/low_mass_u_wt)**u_f
                    if self.ellipse_mode_selector != 'Point Estimates':
                        self.input_data_ellipse['238U/235U c'] = self.input_data_ellipse['238U/235U']*(high_mass_u_wt/low_mass_u_wt)**u_f
                    print('U Bias Calculated by Standard')
                else:
                    u_bias_std = u_bias_type
                    u_bias_std_df = self.input_data[self.input_data['Sample'] == u_bias_type]
                    u_bias_std_df = u_bias_std_df.reset_index(drop=True)
                    accepted_u = u_bias_dict[u_bias_std]['238U/235U']
                    u_f = np.log(accepted_u/np.mean(u_bias_std_df['238U/235U']))/np.log(high_mass_u_wt/low_mass_u_wt)
                    self.input_data['238U/235U c'] = self.input_data['238U/235U']*(high_mass_u_wt/low_mass_u_wt)**u_f
                    if self.ellipse_mode_selector != 'Point Estimates':
                        self.input_data_ellipse['238U/235U c'] = self.input_data_ellipse['238U/235U']*(high_mass_u_wt/low_mass_u_wt)**u_f
                    print('U Bias Calculated Externally')
                    
            df_primary = self.input_data[self.input_data['Sample'] == primary_std]
            df_primary = df_primary.reset_index(drop=True)
            df_secondary = self.input_data[self.input_data['Sample'].isin(secondary_standard_list)]
            df_secondary = df_secondary.reset_index(drop=True)
            
            
            mask = df_secondary['Sample'].isin(stds_dict.keys())
            
            if len(df_secondary[mask]) >= 1:
                for s in df_secondary['Sample'].unique():
                    s_df = df_secondary[df_secondary['Sample'] == s]
                    s_common207206 = stds_dict.get(s)[3] / stds_dict.get(s)[2]
                    s_common207206_error = 1e-50
                    s_df['SK 207Pb/206Pb'] = s_common207206
                    s_df['Common 207Pb/206Pb'] = s_common207206
                    s_df['Common 207Pb/206Pb Error'] = s_common207206_error
                    dummy_df = s_df
                    secondary_ages = calc_fncs.correct_sample_ages(s_df,dummy_df,df_primary,dummy_df,primary_std,ThUzirconratio_stds,ThUmagmaratio_stds,commonPb_Thdisequil_treatment_stds,
                                                                   u_pb_ratio_treatment,'Point Estimates',self.power,UTh_disequil_stds,s_common207206,s_common207206_error,
                                                                   drift_treatment,drift_nearest)
                    if self.output_secondary_data is None:
                        self.output_secondary_data = secondary_ages
                    else:
                        # self.output_secondary_data = self.output_secondary_data.append(secondary_ages, ignore_index=True)
                        self.output_secondary_data = pd.concat([self.output_secondary_data,secondary_ages],ignore_index=True)
                        
            print('DATA UPLOADED')
            fastgrid_layout.close_modal()


    @pn.depends('input_data', 'text_standard_selector', 'output_secondary_data', 'secondary_std_selector',
                'drift_selector', 'drift_nearest_amount', 'drift_analyte_dropdown',
                'mass_bias_pb','mass_bias_thu','regression_selector', 'ellipse_mode_selector',
                'text_standard_selector','ThU_zrn_input','ThU_magma_input','Pb_Th_std_crct_selector','power','UTh_std_norm','common_207206_input','common_207206_error'
                )
    def call_drift_plot(self):
        if (self.ellipse_mode_selector == 'Point Estimates' or self.ellipse_mode_selector == 'Both') and self.text_sample_selector != 'Input Sample ID':
            chosen_std = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_standard_selector)]
            unknown_df = self.input_data[(self.input_data['Sample'] != self.text_standard_selector) & (~self.input_data['Sample'].isin(self.secondary_std_selector)) ]
            return calc_fncs.plot_drift(chosen_std, self.output_secondary_data, self.secondary_std_selector, 
                                        unknown_df, self.drift_analyte_dropdown,
                                        self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                        self.ellipse_mode_selector, self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,
                                        self.drift_selector,self.drift_nearest_amount
                                        )
        else:
            pass

    @pn.depends('input_data', 'input_data_ellipse', 'common_207206_input', 'common_207206_error', 'ellipse_mode_selector', 'text_sample_selector', watch=True)
    def _updateCommonPb(self):
        if self.text_sample_selector != 'Input Sample ID':
            if self.ellipse_mode_selector == 'Point Estimates':
                if self.common_207206_input != 0:
                    self.input_data['Common 207Pb/206Pb'] = self.common_207206_input
                    self.input_data['Common 207Pb/206Pb Error'] = self.common_207206_error
                self.input_data['SK 206Pb/204Pb'] = 11.152 + 9.74*(np.exp(lambda_238*3.7e9)-np.exp(lambda_238*self.input_data['206/238U_age_init']))
                self.input_data['SK 207Pb/204Pb'] = 12.998 + 9.74/137.82*(np.exp(lambda_235*3.7e9)-np.exp(lambda_235*self.input_data['206/238U_age_init']))
                self.input_data['SK 207Pb/206Pb'] = self.input_data['SK 207Pb/204Pb'] / self.input_data['SK 206Pb/204Pb']
            elif self.ellipse_mode_selector == 'Ellipses':
                if self.common_207206_input != 0:
                    self.input_data_ellipse['Common 207Pb/206Pb'] = self.common_207206_input
                    self.input_data_ellipse['Common 207Pb/206Pb Error'] = self.common_207206_error
                self.input_data_ellipse['SK 206Pb/204Pb'] = 11.152 + 9.74*(np.exp(lambda_238*3.7e9)-np.exp(lambda_238*self.input_data_ellipse['206/238U_age_init']))
                self.input_data_ellipse['SK 207Pb/204Pb'] = 12.998 + 9.74/137.82*(np.exp(lambda_235*3.7e9)-np.exp(lambda_235*self.input_data_ellipse['206/238U_age_init']))
                self.input_data_ellipse['SK 207Pb/206Pb'] = self.input_data_ellipse['SK 207Pb/204Pb'] / self.input_data_ellipse['SK 206Pb/204Pb']
            elif self.ellipse_mode_selector == 'Both':
                if self.common_207206_input != 0:
                    self.input_data['Common 207Pb/206Pb'] = self.common_207206_input
                    self.input_data['Common 207Pb/206Pb Error'] = self.common_207206_error
                    self.input_data_ellipse['Common 207Pb/206Pb'] = self.common_207206_input
                    self.input_data_ellipse['Common 207Pb/206Pb Error'] = self.common_207206_error
                self.input_data['SK 206Pb/204Pb'] = 11.152 + 9.74*(np.exp(lambda_238*3.7e9)-np.exp(lambda_238*self.input_data['206/238U_age_init']))
                self.input_data['SK 207Pb/204Pb'] = 12.998 + 9.74/137.82*(np.exp(lambda_235*3.7e9)-np.exp(lambda_235*self.input_data['206/238U_age_init']))
                self.input_data['SK 207Pb/206Pb'] = self.input_data['SK 207Pb/204Pb'] / self.input_data['SK 206Pb/204Pb']
                self.input_data_ellipse['SK 206Pb/204Pb'] = 11.152 + 9.74*(np.exp(lambda_238*3.7e9)-np.exp(lambda_238*self.input_data_ellipse['206/238U_age_init']))
                self.input_data_ellipse['SK 207Pb/204Pb'] = 12.998 + 9.74/137.82*(np.exp(lambda_235*3.7e9)-np.exp(lambda_235*self.input_data_ellipse['206/238U_age_init']))
                self.input_data_ellipse['SK 207Pb/206Pb'] = self.input_data_ellipse['SK 207Pb/204Pb'] / self.input_data_ellipse['SK 206Pb/204Pb']
        print('Common Pb Calculated')
                
                

    @pn.depends('input_data', 'input_data_ellipse', 'text_sample_selector', 'y_axis_TW', 'x_axis_TW', 'label_toggle',
                'x_axis_Weth', 'y_axis_Weth','ellipse_mode_selector', 'power', 'common_207206_input')
    def call_Concordia(self):
        if self.ellipse_mode_selector == 'Point Estimates':
            if self.text_sample_selector != 'Input Sample ID':
                data_toplot = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
                TW_concordia = calc_fncs.plot_TW(data_toplot,0,
                                                 self.x_axis_TW, self.y_axis_TW,
                                                 self.label_toggle, self.ellipse_mode_selector, self.power,self.common_207206_input)
                
                Weth_concordia = calc_fncs.plot_weth(data_toplot,0, 
                                                     self.x_axis_Weth, self.y_axis_Weth,
                                                     self.label_toggle, self.ellipse_mode_selector, self.power)
                tabs = pn.Tabs(('T-W',TW_concordia),('Weth.',Weth_concordia),dynamic=True)
                return tabs

        elif self.ellipse_mode_selector == 'Ellipses':
            if self.text_sample_selector != 'Input Sample ID':
                data_toplot = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_sample_selector)]
                TW_concordia = calc_fncs.plot_TW(data_toplot,0,
                                                 self.x_axis_TW, self.y_axis_TW,
                                                 self.label_toggle, self.ellipse_mode_selector, self.power,self.common_207206_input)
                
                Weth_concordia = calc_fncs.plot_weth(data_toplot,0, 
                                                     self.x_axis_Weth, self.y_axis_Weth,
                                                     self.label_toggle, self.ellipse_mode_selector, self.power)
                tabs = pn.Tabs(('T-W',TW_concordia),('Weth.',Weth_concordia),dynamic=True)
                return tabs
        elif self.ellipse_mode_selector == 'Both':
            if self.text_sample_selector != 'Input Sample ID':
                data_toplot_pts = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
                data_toplot_ell = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_sample_selector)]
                # needs to take both data inputs now
                TW_concordia = calc_fncs.plot_TW(data_toplot_pts,data_toplot_ell,
                                                 self.x_axis_TW, self.y_axis_TW,
                                                 self.label_toggle, self.ellipse_mode_selector, self.power,self.common_207206_input)
                
                Weth_concordia = calc_fncs.plot_weth(data_toplot_pts,data_toplot_ell, 
                                                     self.x_axis_Weth, self.y_axis_Weth,
                                                     self.label_toggle, self.ellipse_mode_selector, self.power)
                tabs = pn.Tabs(('T-W',TW_concordia),('Weth.',Weth_concordia),dynamic=True)
                return tabs
                
                
        else:
            pass
        


    @pn.depends('input_data', 'text_sample_selector', 'text_standard_selector', 'label_toggle', 'ThU_zrn_input', 'ThU_magma_input', 'Pb_Th_std_crct_selector', 'regression_selector',
                'ellipse_mode_selector', 'power', 'UTh_std_norm', 'common_207206_input', 'common_207206_error')
    def call_boxplot(self):
        if (self.ellipse_mode_selector == 'Point Estimates' or self.ellipse_mode_selector == 'Both'):
            if self.text_sample_selector != 'Input Sample ID':
                data_toplot = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
                chosen_std = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_standard_selector)]
                dummy_df = chosen_std
                if self.text_sample_selector == 'Input Sample ID':
                    return 'Placeholder'
                else:
                    ages = calc_fncs.correct_sample_ages(data_toplot, dummy_df, chosen_std, dummy_df, self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                                         'Point Estimates', self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,self.drift_selector,self.drift_nearest_amount)
                    return calc_fncs.plot_boxplot(ages['206Pb/238U_correctedage']/(1e6), ages['SampleLabel'], self.label_toggle, 'Point Estimates')
        else:
            pass

    def add_output_data(self, event=None):
        if self.ellipse_mode_selector == 'Point Estimates':
            data_to_update = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
            chosen_std = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_standard_selector)]
            dummy_df = chosen_std
            ages = calc_fncs.correct_sample_ages(data_to_update, dummy_df, chosen_std, dummy_df, self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                                 self.ellipse_mode_selector, self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,self.drift_selector,self.drift_nearest_amount)
            if self.output_data is None:
                self.output_data = ages
            else:
                # self.output_data = self.output_data.append(ages, ignore_index=True)
                self.output_data = pd.concat([self.output_data,ages],ignore_index=True)
        elif self.ellipse_mode_selector == 'Ellipses':
            chosen_std = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_standard_selector)]
            data_to_update = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_sample_selector)]
            dummy_df = chosen_std
            for i,j in zip(data_to_update.SampleLabel.unique(),data_to_update.measurementindex.unique()):
                data,dataW = calc_fncs.correct_sample_ages(dummy_df,data_to_update[data_to_update['SampleLabel'] == i], dummy_df, chosen_std, self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                                           self.ellipse_mode_selector, self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,self.drift_selector,self.drift_nearest_amount)
                data = pd.concat([data,dataW],axis=1)
                data['measurementindex'] = j
                data['SampleLabel'] = i
                if self.output_data_ellipse is None:
                    self.output_data_ellipse = data
                else:
                    # self.output_data_ellipse = self.output_data_ellipse.append(data, ignore_index=True)
                    self.output_data_ellipse = pd.concat([self.output_data_ellipse,data],ignore_index=True)
        elif self.ellipse_mode_selector == 'Both':
            data_to_update_pts = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
            data_to_update_pts = data_to_update_pts.reset_index(drop=True)
            data_to_update_ell = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_sample_selector)]
            chosen_std = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_standard_selector)]
            chosen_std_ell = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(self.text_standard_selector)]
            ages,dummyell_i,dummyell_iw = calc_fncs.correct_sample_ages(data_to_update_pts, data_to_update_ell, chosen_std, chosen_std_ell, self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                                                        self.ellipse_mode_selector, self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,self.drift_selector,self.drift_nearest_amount)
            if self.output_data is None:
                self.output_data = ages
            else:
                # self.output_data = self.output_data.append(ages, ignore_index=True)
                self.output_data = pd.concat([self.output_data,ages],ignore_index=True)
            # for i,j in zip(data_to_update_ell.SampleLabel.unique(),data_to_update_ell.measurementindex.unique()):
            for i in data_to_update_ell.SampleLabel.unique():
                elldf = data_to_update_ell[data_to_update_ell['SampleLabel'] == i]
                elldf = elldf.reset_index(drop=True)
                data,dataW = calc_fncs.correct_sample_ages(data_to_update_pts.iloc[0],elldf, chosen_std, chosen_std_ell, self.text_standard_selector, self.ThU_zrn_input, self.ThU_magma_input, self.Pb_Th_std_crct_selector, self.regression_selector,
                                                                     'Ellipses', self.power, self.UTh_std_norm,self.common_207206_input,self.common_207206_error,self.drift_selector,self.drift_nearest_amount)
                data = pd.concat([data,dataW],axis=1)
                data['measurementindex'] = elldf['measurementindex'][0]
                data['SampleLabel'] = i
                if self.output_data_ellipse is None:
                    self.output_data_ellipse = data
                else:
                    # self.output_data_ellipse = self.output_data_ellipse.append(data, ignore_index=True)
                    self.output_data_ellipse = pd.concat([self.output_data_ellipse,data],ignore_index=True)

    @pn.depends('output_data', watch=True)
    def _update_data_widget(self):
        if self.output_data is not None:
            self.output_data_widget = self.output_data
            self.output_data_widget.height = 40
            self.output_data_widget.heightpolicy = 'Fixed'
            return pn.widgets.Tabulator(self.output_data_widget, width=800)

    @pn.depends('output_data')
    def export_data(self, event=None):
        if self.ellipse_mode_selector == 'Point Estimates':
            self.output_data.to_excel('output_laserTRAMZ_ages.xlsx')
            self.output_secondary_data.to_excel('output_laserTRAMZ_secondarystd_ages.xlsx')
            
            mask = self.output_secondary_data['Sample'].isin(stds_dict.keys())
            
            if len(self.output_secondary_data[mask]) >= 1:
                for s in self.output_secondary_data['Sample'].unique():
                    s_df = self.output_secondary_data[self.output_secondary_data['Sample'] == s]
                    s_df = s_df.reset_index(drop=True)
                    accepted_206238age = accepted_ages.get(s)[0]
                    accepted_207235age =accepted_ages.get(s)[1]
                    d206238age = s_df['206Pb/238U_correctedage']-accepted_206238age
                    d207235age = s_df['207Pb/235Uc_age']-accepted_207235age
                    
                    fig,ax = plt.subplots(3,2,figsize=(30,30))
                    ax[0,0].plot([min(s_df['235U']),max(s_df['235U'])],[137.818,137.818],'-b',lw=0.5)
                    ax[0,0].errorbar(s_df['235U'],s_df['238U/235U c'],xerr=s_df['235U_1SE']*2,yerr=s_df['SE 238/235']*2,fmt='none',ecolor='k',lw=0.6)
                    ax[0,0].plot(s_df['235U'],s_df['238U/235U c'],'d',mfc='lightgray',mec='k',lw=0)
                    ax[0,0].set_xlabel('CPS 235U')
                    ax[0,0].set_ylabel('238U/235U Corrected')
                    ax[0,0].set_title(s)
                    
                    ax[0,1].errorbar(s_df['207Pb'],s_df['207Pb/206Pb c'],xerr=s_df['207Pb_1SE']*2,yerr=s_df['SE 207/206']*2,fmt='none',ecolor='k',lw=0.6)
                    ax[0,1].plot(s_df['207Pb'],s_df['207Pb/206Pb c'],'d',mfc='lightgray',mec='k',lw=0)
                    ax[0,1].set_xlabel('CPS 207Pb')
                    ax[0,1].set_ylabel('207Pb/206Pb Corrected')
                    
                    ax[1,0].plot([min(s_df['238U/235U c']),max(s_df['238U/235U c'])],[0,0],'-b',lw=0.5)
                    ax[1,0].errorbar(s_df['238U/235U c'],d206238age/1e6,xerr=s_df['SE 238/235']*2,yerr=s_df['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1,0].plot(s_df['238U/235U c'],d206238age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[1,0].set_xlabel('238U/235U Corrected')
                    ax[1,0].set_ylabel('206Pb/238U Age Offset (Ma)')
                    
                    ax[1,1].plot([min(s_df['207Pb/206Pb c']),max(s_df['207Pb/206Pb c'])],[0,0],'-b',lw=0.5)
                    ax[1,1].errorbar(s_df['207Pb/206Pb c'],d206238age/1e6,xerr=s_df['SE 207/206']*2,yerr=s_df['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1,1].plot(s_df['207Pb/206Pb c'],d206238age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[1,1].set_xlabel('207Pb/206Pb Corrected')
                    ax[1,1].set_ylabel('206Pb/238U Age Offset (Ma)')
                    
                    ax[2,0].plot([min(s_df['238U/235U c']),max(s_df['238U/235U c'])],[0,0],'-b',lw=0.5)
                    ax[2,0].errorbar(s_df['207Pb/206Pb c'],d207235age/1e6,xerr=s_df['SE 207/206']*2,yerr=s_df['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[2,0].plot(s_df['207Pb/206Pb c'],d207235age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[2,0].set_xlabel('207Pb/206Pb Corrected')
                    ax[2,0].set_ylabel('207Pb/235U Age Offset (Ma)')
                    
                    ax[2,1].plot([min(s_df['207Pb/206Pb c']),max(s_df['207Pb/206Pb c'])],[0,0],'-b',lw=0.5)
                    ax[2,1].errorbar(s_df['207Pb/206Pb c'],d207235age/1e6,xerr=s_df['SE 207/206']*2,yerr=s_df['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[2,1].plot(s_df['207Pb/206Pb c'],d207235age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[2,1].set_xlabel('207Pb/206Pb Corrected')
                    ax[2,1].set_ylabel('207Pb/235U Age Offset (Ma)')
                    
                    plt.savefig(s+' Age Assessment.png',format='png',dpi=250)

                    boxfig = Figure(figsize=(1, 4))
                    ax = boxfig.add_subplot()

                    bp = ax.boxplot(s_df['206Pb/238U_correctedage']/1e6, patch_artist=True, boxprops=dict(facecolor='slategray', color='k'),
                                    medianprops=dict(color='limegreen'), meanprops=dict(marker='d', mfc='limegreen', mec='k', markersize=4),
                                    flierprops=dict(
                                        marker='o', mfc='None', mec='k', markersize=4),
                                    showmeans=True)

                    ax.text(0.05, 0.8, 'Mean ='+str(round(s_df['206Pb/238U_correctedage'].mean()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.7, 'Med ='+str(round(s_df['206Pb/238U_correctedage'].median()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.6, 'Min ='+str(round(s_df['206Pb/238U_correctedage'].min()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.5, 'Max ='+str(round(s_df['206Pb/238U_correctedage'].max()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.4, 'n = '+str(len(s_df['206Pb/238U_correctedage'])),
                            fontsize=4, transform=ax.transAxes)

                    ax.set_ylabel('206/238 Age (Ma)', fontsize=12)
                    ax.set_xlabel(' ', fontsize=1)
                    ax.tick_params(axis='both', labelsize=8)
                    ax.set_title(s)
                    boxfig.savefig(s+' Boxplot.png',format='png',dpi=250)
                output_secondary_data_grps = self.output_secondary_data.group_by('Sample')
                fig,ax = plt.subplots(2,1,figsize=(8,10))
                ax[0].plot([min(self.output_secondary_data['measurementindex']),max(self.output_secondary_data['measurementindex'])],[0,0],'-b',lw=0.5)
                ax[1].plot([min(self.output_secondary_data['measurementindex']),max(self.output_secondary_data['measurementindex'])],[0,0],'-b',lw=0.5)
                for (i,j),c,m in zip(output_secondary_data_grps,cycle(color_palette),cycle(markers)):
                    accepted_206238age = accepted_ages.get(i)[0]
                    accepted_207235age = accepted_ages.get(i)[1]
                    d206238ages = j['206Pb/238U_correctedage']-accepted_206238age
                    d207235ages = j['207Pb/235Uc_age']-accepted_207235age
                    ax[0].errorbar(j['measurementindex'],d206238ages/1e6,yerr=j['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[0].plot(j['measurementindex'],d206238ages/1e6,marker=m,mfc=c,mec='k',lw=0,label=i)
                    ax[1].errorbar(j['measurementindex'],d207235ages/1e6,yerr=j['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1].plot(j['measurementindex'],d207235ages/1e6,marker=m,mfc=c,mec='k',lw=0,label=i)
                    ax[0].set_xlabel('Measurement')
                    ax[1].set_xlabel('Measurement')
                    ax[0].set_ylabel('206Pb/238U Age Offset (Ma)')
                    ax[1].set_ylabel('207Pb/235U Age Offset (Ma)')
                    ax[0].legend(loc='best')
                plt.savefig('AllSecondary_ages_drift.png',format='png',dpi=250)
                    
                
            
        elif self.ellipse_mode_selector == 'Ellipses':
            self.output_data_ellipse.to_excel(
                'output_lasertramZ_ellipses.xlsx')
        elif self.ellipse_mode_selector == 'Both':
            self.output_data.to_excel('output_laserTRAMZ_ages.xlsx')
            self.output_secondary_data.to_excel('output_laserTRAMZ_secondarystd_ages.xlsx')
            self.output_data_ellipse.to_excel(
                'output_lasertramZ_ellipses.xlsx')
            mask = self.output_secondary_data['Sample'].isin(stds_dict.keys())
            if len(self.output_secondary_data[mask]) >= 1:
                for s in self.output_secondary_data['Sample'].unique():
                    s_df = self.output_secondary_data[self.output_secondary_data['Sample'] == s]
                    s_df = s_df.reset_index(drop=True)
                    accepted_206238age = accepted_ages.get(s)[0]
                    accepted_207235age =accepted_ages.get(s)[1]
                    d206238age = s_df['206Pb/238U_correctedage']-accepted_206238age
                    d207235age = s_df['207Pb/235Uc_age']-accepted_207235age
                    
                    fig,ax = plt.subplots(3,2,figsize=(30,30))
                    ax[0,0].plot([min(s_df['235U']),max(s_df['235U'])],[137.818,137.818],'-b',lw=0.5)
                    ax[0,0].errorbar(s_df['235U'],s_df['238U/235U c'],xerr=s_df['235U_1SE']*2,yerr=s_df['SE 238/235']*2,fmt='none',ecolor='k',lw=0.6)
                    ax[0,0].plot(s_df['235U'],s_df['238U/235U c'],'d',mfc='lightgray',mec='k',lw=0)
                    ax[0,0].set_xlabel('CPS 235U')
                    ax[0,0].set_ylabel('238U/235U Corrected')
                    ax[0,0].set_title(s)
                    
                    ax[0,1].errorbar(s_df['207Pb'],s_df['207Pb/206Pb c'],xerr=s_df['207Pb_1SE']*2,yerr=s_df['SE 207/206']*2,fmt='none',ecolor='k',lw=0.6)
                    ax[0,1].plot(s_df['207Pb'],s_df['207Pb/206Pb c'],'d',mfc='lightgray',mec='k',lw=0)
                    ax[0,1].set_xlabel('CPS 207Pb')
                    ax[0,1].set_ylabel('207Pb/206Pb Corrected')
                    
                    ax[1,0].plot([min(s_df['238U/235U c']),max(s_df['238U/235U c'])],[0,0],'-b',lw=0.5)
                    ax[1,0].errorbar(s_df['238U/235U c'],d206238age/1e6,xerr=s_df['SE 238/235']*2,yerr=s_df['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1,0].plot(s_df['238U/235U c'],d206238age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[1,0].set_xlabel('238U/235U Corrected')
                    ax[1,0].set_ylabel('206Pb/238U Age Offset (Ma)')
                    
                    ax[1,1].plot([min(s_df['207Pb/206Pb c']),max(s_df['207Pb/206Pb c'])],[0,0],'-b',lw=0.5)
                    ax[1,1].errorbar(s_df['207Pb/206Pb c'],d206238age/1e6,xerr=s_df['SE 207/206']*2,yerr=s_df['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1,1].plot(s_df['207Pb/206Pb c'],d206238age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[1,1].set_xlabel('207Pb/206Pb Corrected')
                    ax[1,1].set_ylabel('206Pb/238U Age Offset (Ma)')
                    
                    ax[2,0].plot([min(s_df['238U/235U c']),max(s_df['238U/235U c'])],[0,0],'-b',lw=0.5)
                    ax[2,0].errorbar(s_df['238U/235U c'],d207235age/1e6,xerr=s_df['SE 238/235']*2,yerr=s_df['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[2,0].plot(s_df['238U/235U c'],d207235age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[2,0].set_xlabel('238U/235U Corrected')
                    ax[2,0].set_ylabel('207Pb/235U Age Offset (Ma)')
                    
                    ax[2,1].plot([min(s_df['207Pb/206Pb c']),max(s_df['207Pb/206Pb c'])],[0,0],'-b',lw=0.5)
                    ax[2,1].errorbar(s_df['207Pb/206Pb c'],d207235age/1e6,xerr=s_df['SE 207/206']*2,yerr=s_df['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[2,1].plot(s_df['207Pb/206Pb c'],d207235age/1e6,'d',mfc='lightgray',mec='k',lw=0)
                    ax[2,1].set_xlabel('207Pb/206Pb Corrected')
                    ax[2,1].set_ylabel('207Pb/235U Age Offset (Ma)')
                    
                    plt.savefig(s+' Age Assessment.png',format='png',dpi=250)

                    boxfig = Figure(figsize=(1, 4))
                    ax = boxfig.add_subplot()

                    bp = ax.boxplot(s_df['206Pb/238U_correctedage']/1e6, patch_artist=True, boxprops=dict(facecolor='slategray', color='k'),
                                    medianprops=dict(color='limegreen'), meanprops=dict(marker='d', mfc='limegreen', mec='k', markersize=4),
                                    flierprops=dict(
                                        marker='o', mfc='None', mec='k', markersize=4),
                                    showmeans=True)

                    ax.text(0.05, 0.8, 'Mean ='+str(round(s_df['206Pb/238U_correctedage'].mean()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.7, 'Med ='+str(round(s_df['206Pb/238U_correctedage'].median()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.6, 'Min ='+str(round(s_df['206Pb/238U_correctedage'].min()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.5, 'Max ='+str(round(s_df['206Pb/238U_correctedage'].max()/1e6, 2)),
                            fontsize=4, transform=ax.transAxes)
                    ax.text(0.05, 0.4, 'n = '+str(len(s_df['206Pb/238U_correctedage'])),
                            fontsize=4, transform=ax.transAxes)

                    ax.set_ylabel('206/238 Age (Ma)', fontsize=12)
                    ax.set_xlabel(' ', fontsize=1)
                    ax.tick_params(axis='both', labelsize=8)
                    ax.set_title(s)
                    boxfig.savefig(s+' Boxplot.png',format='png',dpi=250)
                output_secondary_data_grps = self.output_secondary_data.groupby('Sample')
                fig,ax = plt.subplots(2,1,figsize=(8,10))
                ax[0].plot([min(self.output_secondary_data['measurementindex']),max(self.output_secondary_data['measurementindex'])],[0,0],'-b',lw=0.5)
                ax[1].plot([min(self.output_secondary_data['measurementindex']),max(self.output_secondary_data['measurementindex'])],[0,0],'-b',lw=0.5)
                for (i,j),c,m in zip(output_secondary_data_grps,cycle(color_palette),cycle(markers)):
                    accepted_206238age = accepted_ages.get(i)[0]
                    accepted_207235age = accepted_ages.get(i)[1]
                    d206238ages = j['206Pb/238U_correctedage']-accepted_206238age
                    d207235ages = j['207Pb/235Uc_age']-accepted_207235age
                    ax[0].errorbar(j['measurementindex'],d206238ages/1e6,yerr=j['∆206/238 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[0].plot(j['measurementindex'],d206238ages/1e6,marker=m,mfc=c,mec='k',lw=0,label=i)
                    ax[1].errorbar(j['measurementindex'],d207235ages/1e6,yerr=j['∆207/235 age (tot.)']*2/1e6,fmt='none',ecolor='k',lw=0.6)
                    ax[1].plot(j['measurementindex'],d207235ages/1e6,marker=m,mfc=c,mec='k',lw=0,label=i)
                    ax[0].set_xlabel('Measurement')
                    ax[1].set_xlabel('Measurement')
                    ax[0].set_ylabel('206Pb/238U Age Offset (Ma)')
                    ax[1].set_ylabel('207Pb/235U Age Offset (Ma)')
                    ax[0].legend(loc='best')
                plt.savefig('AllSecondary_ages_drift.png',format='png',dpi=250)
            

    def export_plot(self, event=None):
        if self.ellipse_mode_selector == 'Point Estimates':
            data_toplot = self.input_data[self.input_data['SampleLabel'].str.contains(self.text_sample_selector)]
            plot = calc_fncs.plot_TW(data_toplot,
                                     self.x_axis_TW_min, self.x_axis_TW_max,
                                     self.y_axis_TW[0], self.y_axis_TW[1],
                                     self.label_toggle, self.ellipse_mode_selector, self.power)
            plot.savefig('LaserTRAMZ_TW.pdf', format='pdf', dpi=250)
        elif self.ellipse_mode_selector == 'Ellipses':
            data_toplot = self.input_data_ellipse[self.input_data_ellipse['SampleLabel'].str.contains(
                self.text_sample_selector)]
            plot = calc_fncs.plot_TW(data_toplot,
                                     self.x_axis_TW_min, self.x_axis_TW_max,
                                     self.y_axis_TW[0], self.y_axis_TW[1],
                                     self.label_toggle, self.ellipse_mode_selector, self.power)
            plot.savefig('LaserTRAMZ_TW.pdf', format='pdf', dpi=250)


reduce_ages = finalize_ages(name='Reduce Ages')

# %%

pn.extension('tabulator','mathjax')

modal_button_one=pn.WidgetBox(pn.Param(reduce_ages.param.accept_reduction_parameters_button,
                                       widgets={'accept_reduction_parameters_button': pn.widgets.Button(name='Accept Reduction Parameters',button_type='success')}))

widgets={'label_toggle': pn.widgets.CheckBoxGroup,
         'export_data_button': pn.widgets.Button(name='DDDT!', button_type='success'),
         'x_axis_TW':pn.widgets.EditableRangeSlider(name='TW X-lim',start=0,end=100,value=(0,25),step=10),
         'y_axis_TW':pn.widgets.EditableRangeSlider(name='TW Y-lim',start=0,end=1,value=(0,0.3),step=0.1),
         'x_axis_Weth':pn.widgets.EditableRangeSlider(name='Weth X-lim',start=0,end=100,value=(0,40),step=10),
         'y_axis_Weth':pn.widgets.EditableRangeSlider(name='Weth Y-lim',start=0,end=1,value=(0,0.8),step=0.1),
         'regression_selector': pn.widgets.RadioButtonGroup,
         'ellipse_mode_selector': pn.widgets.RadioButtonGroup,
         'Pb_Th_std_crct_selector': pn.widgets.RadioButtonGroup,
         'mass_bias_nist_selector': pn.widgets.RadioButtonGroup,
         'UTh_std_norm': pn.widgets.RadioBoxGroup}


fastgrid_layout = pn.template.VanillaTemplate(title='LaserTRAMZ Concordia: LA-ICP-MC-MS',
                                                sidebar=pn.Column(pn.WidgetBox(pn.Param(reduce_ages.param,widgets=widgets))),sidebar_width=380)

fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())


# fastgrid_layout.main.append(pn.Column(pn.Row(reduce_ages.call_Concordia,reduce_ages.call_boxplot))) # for vanilla
fastgrid_layout.main.append(pn.Row(pn.Column(reduce_ages.call_Concordia),pn.Column(reduce_ages.call_boxplot)))
fastgrid_layout.main.append(pn.Row(reduce_ages.call_drift_plot))
fastgrid_layout.main.append(pn.Column(reduce_ages._update_data_widget)) # for vanilla
fastgrid_layout.show();




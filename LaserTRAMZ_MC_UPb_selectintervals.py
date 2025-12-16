#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:20:07 2025

@author: ctlewis
"""


import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.layouts import row
import panel as pn
import param
from itertools import cycle
import matplotlib as mpl
mpl.use('agg')
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

volt_count_constant = 1.602e-8 #volts / count
color_palette = bokeh.palettes.Muted9

    
        
# %%        
class plots():
    """ Class that holds all of the functions for reducing the time resolved data"""
    def __init__(self,*args):
        super().__init__(*args)
        for a in args:
            self.__setattr__(str(a), args[0])
    
    def simple_ablation_plot(data_ablation_,analysis_start_input,analysis_end_input,logdata,analytes_input):  
        """
        Function to plot the measured isotopic data across the entire analysis (background+ablation+washout)
    
        Parameters
        ----------
        data_ablation_ : pandas dataframe
            dataframe of the measured values to visualize
        analysis_start_input : float
            lower float value of the analysis_start param used to select start of analysis
        bckgrund_stop_input : float
            upper float value of the analysis_end param used to select end of analysis
        logdata: boolean
            allows user to choose if data should be on log scale
        analytes: object
            list of individual analytes to include on ablation plot
    
        Returns
        -------
        fig : bokeh fig
            Figure showing all of the time resolved ablation data
    
        """
        # assign variable y_type to log data or not, based on boolean input
        if logdata == True:
            y_type = 'log'
        else:
            y_type = 'auto'
        # intialize figure
        data_in_frame = data_ablation_[(data_ablation_['Time']>=analysis_start_input) & (data_ablation_['Time']<=analysis_end_input)]
        max_list = []
        min_list = []
        for a in analytes_input:
            max_list.append(max(data_in_frame[a]))
            min_list.append(min(data_in_frame[a]))
        
        min_val = min(min_list)
        max_val = max(max_list)
        fig = figure(height=350,width=1200,title='All Time Resolved Data',tools='pan,reset,save,wheel_zoom,xwheel_zoom,ywheel_zoom',toolbar_location='left',
                      y_axis_type=y_type,x_axis_label = 'Time (s)', y_axis_label = 'Intensities (cps)',
                      x_range=[analysis_start_input-3,analysis_end_input+10],y_range=[min_val-min_val*0.05,max_val+max_val*0.01]
                      )
        var_cols=analytes_input # reassign anlytes_ variable
        # zip colors and analytes together to get plotted vs time
        for i,c in zip(var_cols,cycle(color_palette)):
            fig.line(data_ablation_.Time,data_ablation_[i],line_width=0.7,legend_label='{}'.format(i),color=c)
            
        fig.line([analysis_start_input,analysis_start_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation start
        fig.line([analysis_end_input,analysis_end_input],[min_val-min_val*0.05,max_val+max_val*0.01],line_width=0.4,line_dash='solid',color='black') # vertical solid line for ablation stop
        
        return fig



# %%
class define_analyses(param.Parameterized):
    analytes_ = param.ListSelector(default=[],objects=[],precedence=0.2) # set up empty list to be populated with analytes in order to choose which gets plotted
    logcountsdata = param.Boolean(False,label='Log Intensities',precedence=0.3) # set up boolean button to choose whether or not 
    
    add_reduction_button = param.Action(lambda x: x.add_reduction(), label='Add Reduction',precedence=0) # button that addes current interval (background + ablation) to get reduced. Triggers modal to name analysis
    store_interval_button = param.Action(lambda x: x.store_interval(),label='Store Interval',precedence=1) # button that triggers sample name and selected interval to be stored
    
    jump_sliders_button = param.Action(lambda x: x.jump_sliders(),label='Jump Sliders',precedence=0.1) # button that triggers sliders to jump forward
    jump_time = param.Number(58,precedence=0.15) # number that defines how far to jump sliders when function is triggered
    
    input_data = param.DataFrame(precedence=-1) # initialize dataframe to be populated with uploaded data
    file_path = param.String(default='Insert File Path',precedence=0.7) # string that will be populated with file path
    stored_intervals_data = param.DataFrame(precedence=-1) # initialize dataframe to be populated with output data
    integration_time = param.Number(default=0.01,precedence=-1)
    
    analysis_start = param.Number(23.4,bounds=(0,8600),softbounds=(50,90),step=0.1,precedence=0.9) # number that defines where analysis interval starts
    analysis_end = param.Number(77.3,bounds=(0,8600),step=0.1,precedence=0.8) # number that defines where analysis interval ends
    
    accept_array_button = param.Action(lambda x: x.accept_array(),label='Accept Detector Array',precedence=1) # button that triggers the collector block anaalyte assignments to be accepted
    
    def __init__(self,**params):
        super().__init__(**params)
        self.file_input_widget = pn.Param(self.param.input_data)
        self.stored_intervals_widget = pn.Param(self.param.stored_intervals_data)
        self.widgets = pn.Param(self,parameters=['accept_array_button','analytes_',
                                                 'add_reduction_button','store_interval_button',
                                                 'jump_sliders_button','jump_time',
                                                 'file_path','integration_time',
                                                 'analysis_start','analysis_end','logcountsdata'
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
                # loop through rows and columns, putting in a string and button to select if data is incoming as volts or counts
                fastgrid_layout.modal[rth_row].append(pn.Column(pn.Row(pn.widgets.TextInput(placeholder='Mass')),
                                                                pn.Row(pn.widgets.RadioButtonGroup(options=['Volts','Counts']))))
                nth_iter = nth_iter + 1
                if nth_iter % 3 == 0:
                    rth_row = rth_row + 1
            fastgrid_layout.modal[-1].append(pn.Column(buttons_))
            fastgrid_layout.open_modal()
            
    @pn.depends('analytes_')
    def accept_array(self,event=None):
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
            if detector_type == 'Volts':
                self.input_data[next_analyte] = self.input_data[next_analyte] / volt_count_constant
            elif detector_type == 'Counts':
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
        self.stored_intervals_data = pd.DataFrame([np.zeros(len(self.input_data.columns))],columns=list(self.input_data.columns))
        self.stored_intervals_data.insert(0,'measurementindex',0)
        self.stored_intervals_data.insert(1,'SampleLabel',0)
        self.stored_intervals_data.insert(2,'Analysis Start',0)
        self.stored_intervals_data.insert(3,'Analysis End',0)

        fastgrid_layout.close_modal()
            
            
    @pn.depends('analysis_start','analysis_end')
    def jump_sliders(self,event=None, watch=True):
        """
        Function that advances ('jumps') sliders forward based on the jump time

        """
        self.analysis_end = self.analysis_end+self.jump_time
        self.analysis_start = self.analysis_start+self.jump_time
        
        
    @pn.depends('input_data','analysis_start','analysis_end','analytes_','logcountsdata')
    def call_simple_ablation_plot(self):
        """
        Function that calls and places the plot with time resolved analytes into a bokeh pane

        Returns
        -------
        bokeh pane
            hosts figure

        """
        if self.stored_intervals_data is not None:
            data_toplot = self.input_data
            return pn.pane.Bokeh(row(plots.simple_ablation_plot(data_toplot,self.analysis_start,self.analysis_end,self.logcountsdata,self.analytes_)))
        
        
    @pn.depends('input_data')
    def add_reduction(self, event=None):
        """
        Function that clears any residuals input from uploading data or previously approved analyses, then generates fresh ones in a modal

        Parameters
        ----------
        event : open panel modal

        """      
        # clear current modal
        fastgrid_layout.modal[0].clear()
        fastgrid_layout.modal[1].clear()
        fastgrid_layout.modal[2].clear()
        fastgrid_layout.modal[3].clear()
        fastgrid_layout.modal[4].clear()
        
        # put button and text box on modal for recording sample name
        fastgrid_layout.modal[0].append(store_interval_widget_button) # this needs to be a button widget implemented when calling class instance that is linked to the store_interval_button above
        fastgrid_layout.modal[1].append(pn.widgets.TextInput(placeholder='Enter Sample Name'))
        
        fastgrid_layout.open_modal()
        
        
    def store_interval(self,event=None):
        """
        Function that gets the fully reduced data and sends it to the output data file that will be exported. Closes modal.

        """
        new_interval_df = pd.DataFrame([np.zeros(len(self.stored_intervals_data.columns))],columns=self.stored_intervals_data.columns)
        sample_name = fastgrid_layout.modal[1][0].value
        new_interval_df['SampleLabel'] = sample_name
        new_interval_df['Analysis Start'] = self.analysis_start
        new_interval_df['Analysis End'] = self.analysis_end

            
        self.stored_intervals_data = pd.concat([self.stored_intervals_data,new_interval_df],ignore_index=True)
    
        fastgrid_layout.close_modal()
        
        
    @pn.depends('stored_intervals_data',watch=True)
    def _update_stored_intervals_widget(self):
        """
        Function that displays intervals data when updated

        Returns
        -------
        Tabulator table
            hosts output data

        """
        if self.stored_intervals_data is not None:
            self.stored_intervals_widget = self.stored_intervals_data
            self.stored_intervals_widget.height = 400
            self.stored_intervals_widget.heightpolicy = 'Fixed'
            return pn.widgets.Tabulator(self.stored_intervals_widget,width=600) # use 600 for large screen, 100-150 for small screen
    
    @pn.cache(per_session=True)
    def share_data(self):
        if self.input_data is None:
            pn.state.cache['input_data'] = self.input_data
        if self.stored_intervals_data is None:
            pn.state.cache['stored_intervals_data'] = self.stored_intervals_data
            
        return pn.state.cache['input_data'], pn.state.cache['stored_intervals_data']
            
        
callapp = define_analyses(name='Define Analysis Intervals')

# %%


pn.extension('tabulator','mathjax')

buttons_=pn.WidgetBox(pn.Param(callapp.param.accept_array_button,widgets={'accept_array_button': pn.widgets.Button(name='Accept Detector Array',button_type='success')}))
store_interval_widget_button=pn.WidgetBox(pn.Param(callapp.param.store_interval_button,widgets={'store_interval_button': pn.widgets.Button(name='Store Interval',button_type='success')}))

widgets={'analytes_': pn.widgets.CheckBoxGroup,
         'analysis_start': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(23.4),step=0.1,name='Ablation Start'),
         'analysis_end': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(77.3),step=0.1,name='Ablation End'),
         }

    
fastgrid_layout = pn.template.VanillaTemplate(title='LaserTRAMZ: LA-MC-ICP-MS Select Intervals',
                                              sidebar=pn.Column(pn.WidgetBox(pn.Param(callapp.param,widgets=widgets))),sidebar_width=380)

fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())
fastgrid_layout.modal.append(pn.Row())

fastgrid_layout.main.append(pn.Row(pn.WidgetBox(pn.Param(callapp.param.analysis_start,widgets={'analysis_start': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(51.1),step=1,name='Analysis Start',width=1300,height=20)},),
                                                pn.Param(callapp.param.analysis_end,widgets={'analysis_end': pn.widgets.EditableFloatSlider(start=0,end=8600,value=(76.3),step=1,name='Analysis End',width=1300,height=20),}),
                                                width=1400,height=180
                                                )
                                   )
                            ) # for vanilla

fastgrid_layout.main.append(pn.Column(callapp.call_simple_ablation_plot)) # for vanilla
fastgrid_layout.main.append(pn.Column(callapp._update_stored_intervals_widget))

fastgrid_layout.show();
# fastgrid_layout.servable()




    

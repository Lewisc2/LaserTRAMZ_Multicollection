# LaserTRAMZ LA-ICP-MC-MS
____________
## Citing and Related Documentation
#### See citation.cff file for citing information
____________
## Purpose
#### Open-source software built to reduce and handle time-resolved U-Pb data collected on a Nu multicollector ICP-MS. Two primary components are in this software. The first reduces the raw data. The second takes the reduced data and calculates ages.
____________
## Data Input Format
**Part One**
#### Convert the .csv file output from the MS into a .xlsx file. It should look like this:
![alt text](LaserTRAMZ_inputfile.png)

**Part Two**
#### The first column and first row of the output file from part one should be deleted. Standards will need to have a required name in order to be recognized. Currently these are all zircon reference materials from the PlasmAge consortium or NIST glasses. If you have a standard that you want put in send an email. See below for input names, which will will demark in Part one as highlighted below.
| Standard          | Reference                 | Input Name Required |
| ----------------- | ------------------------- | ------------------- |
| Fish Canyon Tuff  | Schmitz and Bowring, 2001 | FishCanyon          |
| 94-35             | Klepeis et al., 1998      | 94-35               |
| Plešovice         | Sláma et al., 2008        | Plesovice           |
| Temora2           | Black et al., 2004        | Temora              |
| R33               | Black et al., 2004        | R33                 |
| 91500             | Wiedenbeck et al., 1995   | 91500               |
| FC-1              | Paces and Miller, 1993    | FC1                 |
| Oracle            | Bowring, unpublished      | Oracle              |
| Tan-BrA           | Pecha, unpublished        | Tan-Bra             |
| OG-1              | Stern et al., 2009        | OG1                 |
| NIST-614          | Woodhead and Hergt, 2007  | NIST-614            |
| NIST-612          | Woodhead and Hergt, 2007  | NIST-612            |
| NIST-610          | Woodhead and Hergt, 2007  | NIST-610            |
____________
## Part One: Analyte Reduction
#### Either run the script through the terminal or through an IDE (see more detail below). Input the file path to your file in the 'File path' input. The number of active detectors should automatically be picked up and a window with the appropriate number of possible inputs should pop up. From top left to bottom right, as if you were reading a book, input the analytes in the format MassElement (e.g., 238U). There is an option for each analyte to assign the mass to a Faraday or an Ion Counter. In reality, this really just handles counts to turn them into volts or vice versa in the software. On the Nu P3D all output columns should be in volts and thus all analytes should stay assigned as Faraday. If your mass spec is set to spit out numbers in counts, you will want to pick IC for those analytes. An example of the window is here:
![alt text](LaserTRAMZ_partI_collectorinput.png)
#### Once the analytes are input, click Accept Detector Array. To load the plots onto the screen, click any key on the sliders. You should get a screen that looks something like this without all the colored/labeled squares:
![alt text](LaserTRAMZ_partIscreenshot.png)
#### Labeled squares are as follows: A) GUI tools, B) Plot of isotope ratios, C) Residuals for regressions, D) Intensities (cps) E) Confidence ellipsoids, F) Regression Statistics. More detail on each of these is immediately below.

#### A) Gui Tools. 
* Approve Interval Button: Approve interval sends the current data in the background and ablation intervals to get reduced. When this button is clicked a window pops up for you to input a sample name (I typically print the list off the laser). Once the sample name is input click Accept Sample Name. Reduced data should get populated in the table at the bottom of the screen. Note that the analysis number for that sample and the global analysis number is automatically added. If you skip any analyses you will need to manually update the global analysis number (the table is editable).
* DDDT! Button: This outputs the reduced data #iykyk
* Jump Slider Button: Advances sliders according to jump time (see below)
* Ablation start/end, background start/end, and ablation start true are sliders used to choose the ablation interval. These are plotted as solid, dashed, and dotted lines respectively. You may either slide the sliders, click the arrows, or click and manually input numbers above each of the sliders. Ablation start true is where the regression gets projected back to for the t0 intercept following Kosler et al. (2002). 
* Jump time: Amount to jump the sliders forward. Makes it easy to advance everything. Should be roughly your background+ablation+washout time. WARNING: If you jump very far (e.g., reloading the program after coming back to it and going 5000 secs in the future) the calculations in the background will become very cumbersome for your computer to run due to the crazy values and large time domain the calculations are seeing. You can prevent this by turning off all the ratio buttons, regression buttons, and analyte buttons. It is mostly the ratio buttons and regression buttons that matter.
* List of ratios: Choose which ratios to plot in the ratio plot
* Regression Buttons: Choose whether to include 1st order regression or exponential regression in the reduction. Note at least one needs to be on in order to visualize residuals. We have found that even in total counts mode (see below) it is best to visualize these.
* Log Intensities: Logs intensities in the intensities plot
* Analyte Buttons: Choose which analytes to plot in the intensities plot
* Total Counts / Means and Regression Button: Choose whether to reduce the data using the total counts approach (i.e., integrating the signals, e.g., Johnston et al., 2004) or if Pb-U ratios should be regressed to the time zero intercept.
* Integration time: Integration time in your output file. DO NOT CHANGE THIS
* Generate Ellipse: Whether or not to output ellipsoid data (i.e., all integrations in the ablation interval)
* Power: Alpha value in the confidence ellipse. For 95% confidence leave this at 0.05
* File path: Input file path here
* Extra Buttons: In case you accidentally click out of the pop-up windows but had everything input already, you can run the rest of the calculations that would have been done by clicking the corresponding buttons.
#### B) Ratio Plots: This is a bokeh plot so you can pan by clicking in dragging, zoom with the mouse wheel, reset, or download the plot. Buttons at the top of the plot toggle the options.
#### C) Residuals: Residuals for the regressions. Also a Bokeh plot
#### D) Intensity Plot: Shows time resolved intensities for relevant analytes. Also a bokeh plot.
#### E) Confidence ellipse plots. Click the TW tab at the top for Tera-Wasserburg projection. Weth. for wetherhill projection
#### F) Regression statistics. Note the screenshot may be outdated.
____________
## Part Two: Age Reduction
#### Open the output files from part one and delete the first column and first row. Run the part two script through the terminal or an IDE. Copy and paste the file path into the appropriate input in the GUI tools. You should get a pop up window that looks like:
![alt text](LaserTRAMZ_Optionsinput.png)
#### Input Options are in the order that follows:
* Primary Standard: Start typing the standard you wish to use, which should have a name the same as the table above. You should get a dropdown that has the standard name.
* List of buttons with sample names: Check those you want to use as validation standards. You need at least one for the program to even work
* 1st Order, etc: How you wish to handle the U-Pb ratios. 1st order is 1st order regression, Exp is exponential regression, Total counts is integrated signals
* Point Estimates, etc.: Reduce only point estimates, only confidence ellipsoids (why?), or both. If you choose Ellipses or Both you need to input the file path to the confidence ellipsoid output from part one in the file input immediately below.
* By Age, etc: Pb-Pb mass bias correction standard. If by Age, uses primary ZRM. Currently uses exponential correction of Woodhead (2002)
* Pb-Pb ratios list: Which measured Pb-Pb ratio to use for correcting Pb-Pb mass bias
* Primary 238U/235U, ...: Which standard to use for correcting the 238/235 mass bias. Primary 238U/235U uses primary ZRM and assuems 137.818 (Hiess et al., 2012)
* 238U/235U: Which ratio to use for correcting 38/35 bias. Currently only allows 38/35
* By Age, None: How to correct for Pb-U mass bias. Not sure why you would pick none unless you were testing your raw values for something
* Nearest Number: Nearest number of standards to use for correcting data. I.e., this is the sliding window of Gehrels et al. (2008)
* Common Pb, Common Pb + Th. Disequil: Whether to correct just for Common Pb or to correct for Common Pb and Th Diseuqilibrium
* Calc U/Th, Off: If you've recalculated the U/Th ratio in your zircons from a prior age calculation you may leave this on to use for the Th disequilibrium correction. NOTE: You must have a column in your dataframe with the title *238U/232Th_calc* for this to work. Additionally, this will not work for standards, as they have accepted values or assumed values from prior authors
* Th/U Zircon: Th/U ratio to use in zircon for Th disequilibrium correction
* Th/U Magma: Th/U ratio to use for host melt for Th disequilbrium correction
* Accept Reduction Parameters: Click this value to accept all your inputs
#### Once the parameters are accepted, input a sample name in the Text Sample Selector slot in the Gui tools. You should get a screen that looks like:
![alt text](LaserTRAMZ_partIIscreenshot.png)
#### Labeled squares are as follows: A) GUI tools, B) Concordia, C) Boxplot of ages for current sample, D) Sliding window fractionation factors on standards. An output table with reduced is also at the bottom. More detail on each of these is immediately below.

#### A) Gui Tools
* Drift analyte dropdown: What fractionation factor to visualize in the drift plot
* TW X-lim through Weth Y-lim: Sliders used to control x and y axes on Tera-Wasserburg and Wetherhill concordia
* Text sample selector: Input name of standard you want to reduce
* Common 207206 input / error: If you have projected the data to common Pb (automatic determination incoming at later update) or have measured the common Pb in feldspars, you can input the values and the errors here
* ThU zrn input / ThU magma input: Values to use for Th/U in zircon and Th/U in magma for the Th disequilibrium correction.
* Calc U/Th vs Off: See above. Must have the a column with the name *238U/232Th_calc* for this to work.
* Common Pb / Common Pb + Th Diseuqil: As above
* Power: Alpha value on the confidence ellipsoid
* Approve Data button: Reduce ages for the current sample input
* DDDT!: Output reduced data #iykyk
* Accept Reduction Parameters: If you accidentally clicked outside the popup but had everything input, click this button and it will run the necessary stuff in the background
____________
## Age and Error Calculation Additional Notes
#### Age calculations largely follow prior authors (Kosler and Sylvester, 2003; Pullen et al., 2018; Horstwood et al. (2016)). Decay constants are from Jaffey et al. (1971) (238U / 235U), Cheng et al. (2000) (230Th), and Le Roux and Glendenin (1963) following the recommendations of Steiger and Jager (1977) (232Th). Errors on decay constants are from those authors or, for U, include the uncertainty for counting statistics (238U: 0.16%, 235U: 0.21%; Mattinson (187)) Data points are plotted on Tera-Wasserburg concordia and projected down from common Pb. If you have no common Pb input then the Stacey-Kramers model is assumed. Error propagation largely follows the community accepted protocol (Horstwood et al., 2016) except that we add the uncertainty from the ID-TIMS determinations and omit the excess variance, as we don't know what that value is for your lab. If you wish to include excess variance from the standards in your lab, you will need to input this additional error manually. All necessary error contributions are in the output file (see below). 
#### Mass bias calculations use the exponential fractionation law. Pb-Pb ratios in NIST are from Woodhead and Hergt (2001). 238U/235U ratios in NIST are from Duffin et al. (2015). Masses on isotopes are from their long-time accepted values.
____________
## Final File Output Information
#### You should get a file output with reduced standard ages and reduced unknown ages in two excel workbooks. Plots of standard analyses will also be output. If you choose to reduce ellipsoids a third file will be output with ellipsoid parameters. Some of these are self explanatory but information on what is in these workbooks is as follows:
**Standard and Unknown Ages**
* measurement index: Global measurement number
* SampleLabel: Input sample label and corresponding sample measurement number
* t start - b end: ablation start time, ablation end time, time-zero intercept time, background start time, background end time
* 238U - 204Hg_1SE: Background and interference corrected mean intensities and 1 standard error on all masses. bdl if bdl
* 206Pb/238U - SE% 207/204: Background and interference corrected ratios and 1 standard error. If regressions were chosen these will be in here as well.
* 206Pb/238U_unc / 206/238 Reg. err / 238/206 err: Uncorrected 206/238 ratio, 206/238 error, and 238/206 error from whichever Pb-U treatment you went with. Should match one of the columns to the left. Even in total counts mode this is the error. Column header for error will likely change in later update
* 207/235 Reg. err: Error for the 207/235 ratio from whichever Pb-U treatment you went with.
* 206/238U_age_init / 207/235U_age_init: Initial guess at the 206/238 and 207/235 age to calculate the common Pb ratio from the Stacey-Kramers model. 
* 207Pb/206Pb c: If you measured a NIST glass to characterize the mass bias on Pb-Pb ratios directly, this is the mass bias corrected ratio. If you choose to correct 'By Age' (i.e., by using a ZRM) the mass bias isn't corrected until the age is calculated and this will equal the measured ratio to the left.
* 238U/235U c: Mass Bias corrected 38/35 ratio
* SK 206Pb/204Pb - SK 207Pb/206Pb: Common Pb ratios from the Stacey-Kramers model
* frac_factor_206238 / frac_factor_207235: fractionation factor on the respective ratios calculated from the sliding window
* tims_age_std - tims_error_std_207: Accepted ages and errors on the primary standard from ID-TIMS. See references in table above.
* avg_std_ratio - avg_reg_err_207: Weighted means and errors for the 206/238 ratios and 207/235 ratios from the primary standard. Will change value based on which standards were captured in the sliding window. If you choose to correct for Th disequilibrium the Th corrected aratio will be in these columns with the suffix *_Thcrct*
* 238U/206Pb_corrected / 207Pb/235U_corrected: Mass bias corrected Pb-U ratios according to Kosler and Sylvester (2003)
* 207Pb/206Pbr: Radiogenic 207Pb/206Pb ratio from the concordant projection
* f: fraction of common Pb
* counts_pb206r: Counts of 206Pb that are actually radiogenic in the analysis
* 206Pb/238Upbc_numerical - 207Pb/235Upbc_corrected: Common Pb and mass bias corrected ratios. There are two for 206/238, one is numerically calculated and the other is from the concordant point. They are both in for testing during future updates.
* 207Pb/235Uc_age / ∆207/235 age (tot.): Corrected age and error for the 207/235 ratio. Error propagation and age calculation above.
* 206Pb/238UPbThc: If you choose to correct for Th disequilibrium, this is the mass bias corrected, Th disequilibrium, common Pb corrected 206/238 ratio. 
* 206Pb/238U_correctedage: This is the fully corrected age. Uses whichever ratio is the 'most' corrected based on user options
* ∆206/238 age (meas.) / ∆206/238 age (tot.): 1𝜎 errors for the measurement (206/238 & 207/206 ratios) and the fully propagated error which should be used for reporting.


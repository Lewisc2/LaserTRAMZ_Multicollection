# LaserTRAMZ LA-MC-ICP-MS
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
<img width="602" alt="LaserTRAMZ_inputfile" src="https://github.com/Lewisc2/LaserTRAMZ_Nu/assets/65908927/d1d5d481-c961-4a19-991a-3f08461c79bc">

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
![LaserTRAMZ_partI_collectorinput](https://github.com/Lewisc2/LaserTRAMZ_Nu/assets/65908927/2107416e-7292-44f1-963f-c2b1f637fe1a)
#### Once the analytes are input, click Accept Detector Array. To load the plots onto the screen, click any key on the sliders. You should get a screen that looks something like:
<img width="1708" alt="Screenshot 2025-03-20 at 3 26 10 PM" src="https://github.com/user-attachments/assets/930097cb-0cb2-433b-8edb-6826d6eb059a" />
#### The top graph shows the intensities for each analyte. The bottom left graph shows the time resolved ratios selected in the left banner. The bottom right graph will always only show the 207Pb/206Pb ratio. Note that the plots are interactable (e.g., panning, zooming, downloading, etc). Buttons at the top of the plot toggle the interaction options.

#### Gui Tools. 
* Ablation start/end and background start/end sliders are at the top of the screen. These are plotted as solid and dashed lines, respectively. You may either slide the sliders, click the arrows, or highlight and manually input numbers above each of the sliders.
* Ablation start true is where the regression gets projected back to for the t0 intercept following Kosler et al. (2002). If you wish this to be at the start of hte ablation interval (as opposed to projecting backwards to the true start) you can simply click the 'Lock Back Projection' button. In my experience this is generally more reliable.
* Jump Slider Button: Advances all sliders according to jump time (see below)
* Jump time: Amount to jump the sliders forward. Makes it easy to advance everything for data collected in TRA mode (written specifically for Nu instruments). Should be roughly your background+ablation+washout time.
* List of ratios: Choose which ratios to plot in the ratio plot
* Regression Buttons: Choose whether to include 1st order regression or exponential regression in the reduction. Note at least one needs to be on in order to visualize residuals. Even in total counts mode (see below) it is best to visualize these and so they are always displayed.
* Log Intensities: Logs intensities in the intensities plot
* Analyte Buttons: Choose which analytes to plot in the intensities plot
* Total Counts / Means and Regression Button: Choose whether to reduce the data using the total counts approach (i.e., integrating the signals, e.g., Johnston et al., 2004) or if Pb-U ratios should be regressed to the time zero intercept.
* Evalute Interval Button: Pulls up a window that allows you to visualize the fitted regressions, residuals, and confidence ellipsoid.
* DDDT! Button: This outputs the reduced data
* Integration time: Integration time in your output file. DO NOT CHANGE THIS UNLESS NEEDED. Note: This software currently does not deal with multiple integration time settings, but this feature is coming soon.
* Power: Alpha value in the confidence ellipse. For 95% confidence leave this at 0.05
* File path: Input file path here
* Extra Buttons: In case you accidentally click out of the pop-up windows but had everything input already, you can run the rest of the calculations that would have been done by clicking the corresponding buttons.
#### Evalute Interval - Something akin to the following will pop-up upon clicking the Evaluate Interval Button.
<img width="1330" alt="Screenshot 2025-03-20 at 3 27 35 PM" src="https://github.com/user-attachments/assets/14413aa0-a5ed-4d17-aa18-132ae45a4528" />
* Everything here is static and can not be changed. If you are happy with the ablation interval, type the sample name into the box at the top and click the Accept button. The software will automatically recognize sample names that have previously been input and will autonumber the samples for you (e.g., 91500-1, 91550-2; you only type 91500).
* Residuals plot is also interactable
* On the Confidence ellipse plots click the TW tab at the top for Tera-Wasserburg projection. Weth. for wetherhill projection
C) Residuals: Residuals for the regressions. Also a Bokeh plot

____________
## Part Two: Age Reduction
#### Open the output files from part one and delete the first column and first row. Run the part two script through the terminal or an IDE. Copy and paste the file path into the appropriate input in the GUI tools. You should get a pop up window that looks like:
![LaserTRAMZ_Optionsinput](https://github.com/Lewisc2/LaserTRAMZ_Nu/assets/65908927/9cec1d7b-ac82-4e18-949a-887367b85f59)
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
![LaserTRAMZ_partIIscreenshot](https://github.com/Lewisc2/LaserTRAMZ_Nu/assets/65908927/85e0b0d6-a3b8-48c7-bf86-ae17e8982662)
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
#### B) Concordia Plot: Default to Tera-Wasserburg. Click the Weth. tab for Wetherhill. Control x and y axis extent with sliders in GUI tools
#### C) Boxplot: Has all ages for current sample. Basic statistics are shown in the plot.
#### D) Sliding window fractionation: Shows how the fractionation factor changes throughout the run if you choose to correct for drift. Also see Gehrels et al. (2008) and Pullen et al. (2018).
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
____________
## Installation and Use
#### We recommend running the program through [Anaconda](https://www.anaconda.com/download). You may also need to download [Git](https://github.com/git-guides/install-git).  After downloading, one may clone the repository by opening their terminal or Anconda prompt and running the following lines (one by one). It is best to create a virtual environment, which is included in the code block below:
```
git clone https://github.com/Lewisc2/LaserTRAMZ_Nu.git
cd /path/to/LaserTRAMZ
conda create -n LaserTRAMZ python==3.9.18
pip install -r requirements.txt
```
#### where /path/to/LaserTRAMZ is the file path to the cloned repository. You will need to accept the install by typing y then pressing enter when prompted. Once this is complete, the program can be run by opening Spyder from the Anaconda navigator and running the scripts, or,
```
cd /path/to/LaserTRAMZ_Nu
conda activate LaserTRAMZ_Nu
python LaserTRAMZ_Nu_UPbprogram.py
```
#### for part one and 
```
cd /path/to/LaserTRAMZ
conda activate LaserTRAMZ
python LaserTRAMZ_Nu_UPbprogram_Concordia.py
```
#### for part two
#### To shut down the virtual environemnt, run the following
```
conda deactivate LaserTRAMZ_Nu
```
____________
## Citations
```
Black, L.P. et al., 2004, Improved 206Pb/238U microprobe geochronology by the monitoring of a trace-element-related matrix effect; SHRIMP, ID–TIMS, ELA–ICP–MS and oxygen isotope documentation for a series of zircon standards: Chemical Geology, v. 205, p. 115–140, doi:10.1016/j.chemgeo.2004.01.003.
Cheng, H., Edwards, R.L., Hoff, J., Gallup, C.D., Richards, D.A., and Asmerom, Y., 2000, The half-lives of uranium-234 and thorium-230: Chemical Geology, v. 169, p. 17–33, doi:10.1016/S0009-2541(99)00157-6.
Gehrels, G.E., Valencia, V.A., and Ruiz, J., 2008, Enhanced precision, accuracy, efficiency, and spatial resolution of U-Pb ages by laser ablation-multicollector-inductively coupled plasma-mass spectrometry: TECHNICAL BRIEF: Geochemistry, Geophysics, Geosystems, v. 9, p. n/a-n/a, doi:10.1029/2007GC001805.
Hiess, J., Condon, D.J., McLean, N., and Noble, S.R., 2012, 238U/235U Systematics in Terrestrial Uranium-Bearing Minerals: v. 335.
Horstwood, M.S.A. et al., 2016, Community‐Derived Standards for LA ‐ ICP ‐ MS U‐(Th‐)Pb Geochronology – Uncertainty Propagation, Age Interpretation and Data Reporting: Geostandards and Geoanalytical Research, v. 40, p. 311–332, doi:10.1111/j.1751-908X.2016.00379.x.
Jaffey, A.H., Flynn, K.F., Glendenin, L.E., Bentley, W.C., and Essling, A.M., 1971, Precision Measurement of Half-Lives and Specific Activities of U 235 and U 238: Physical Review C, v. 4, p. 1889–1906, doi:10.1103/PhysRevC.4.1889.
Johnston, S., Gehrels, G., Valencia, V., and Ruiz, J., 2009, Small-volume U–Pb zircon geochronology by laser ablation-multicollector-ICP-MS: Chemical Geology, v. 259, p. 218–229, doi:10.1016/j.chemgeo.2008.11.004.
Klepeis, K.A., Crawford, M.L., and Gehrels, G., 1998, Structural history of the crustal-scale Coast shear zone north of Portland Canal, southeast Alaska and British Columbia: Journal of Structural Geology, v. 20, p. 883–904, doi:10.1016/S0191-8141(98)00020-0.
Košler, J., Fonneland, H., Sylvester, P., Tubrett, M., and Pedersen, R.-B., 2002, U–Pb dating of detrital zircons for sediment provenance studies—a comparison of laser ablation ICPMS and SIMS techniques: Chemical Geology, v. 182, p. 605–618, doi:10.1016/S0009-2541(01)00341-2.
Košler, J., and Sylvester, P.J., 2003, Present Trends and the Future of Zircon in Geochronology: Laser Ablation ICPMS: Reviews in Mineralogy and Geochemistry, v. 53, p. 243–275, doi:https://doi.org/10.2113/0530243.
Mattinson, J.M., 1987, UPb ages of zircons: A basic examination of error propagation: Chemical Geology: Isotope Geoscience section, v. 66, p. 151–162, doi:10.1016/0168-9622(87)90037-6.
Paces, J.B., and Miller, J.D., 1993, Precise U‐Pb ages of Duluth Complex and related mafic intrusions, northeastern Minnesota: Geochronological insights to physical, petrogenetic, paleomagnetic, and tectonomagmatic processes associated with the 1.1 Ga Midcontinent Rift System: Journal of Geophysical Research: Solid Earth, v. 98, p. 13997–14013, doi:10.1029/93JB01159.
Pullen, A., Ibáñez-Mejia, M., Gehrels, G.E., Giesler, D., and Pecha, M., 2018, Optimization of a Laser Ablation-Single Collector-Inductively Coupled Plasma-Mass Spectrometer (Thermo Element 2) for Accurate, Precise, and Efficient Zircon U-Th-Pb Geochronology: Geochemistry, Geophysics, Geosystems, v. 19, p. 3689–3705, doi:10.1029/2018GC007889.
Schmitz, M.D., and Bowring, S.A., 2001, U-Pb zircon and titanite systematics of the Fish Canyon Tuff: an assessment of high-precision U-Pb geochronology and its application to young volcanic rocks: Geochimica et Cosmochimica Acta, v. 65, p. 2571–2587, doi:10.1016/S0016-7037(01)00616-0.
Sláma, J. et al., 2008, Plešovice zircon — A new natural reference material for U–Pb and Hf isotopic microanalysis: Chemical Geology, v. 249, p. 1–35, doi:10.1016/j.chemgeo.2007.11.005.
Steiger, R.H., and Jäger, E., 1977, Subcommission on geochronology: Convention on the use of decay constants in geo- and cosmochronology: Earth and Planetary Science Letters, v. 36, p. 359–362, doi:10.1016/0012-821X(77)90060-7.
Stern, R.A., Bodorkos, S., Kamo, S.L., Hickman, A.H., and Corfu, F., 2009, Measurement of SIMS Instrumental Mass Fractionation of Pb Isotopes During Zircon Dating: Geostandards and Geoanalytical Research, v. 33, p. 145–168, doi:10.1111/j.1751-908X.2009.00023.x.
Wiedenbeck, M., Allé, P., Corfu, F., Griffin, W.L., Meier, M., Oberli, F., Quadt, A.V., Roddick, J.C., and Spiegel, W., 1995, THREE NATURAL ZIRCON STANDARDS FOR U-TH-PB, LU-HF, TRACE ELEMENT AND REE ANALYSES: Geostandards and Geoanalytical Research, v. 19, p. 1–23, doi:10.1111/j.1751-908X.1995.tb00147.x.
Woodhead, J., 2002, A simple method for obtaining highly accurate Pb isotope data by MC-ICP-MS: Journal of Analytical Atomic Spectrometry, v. 17, p. 1381–1385, doi:10.1039/b205045e.
Woodhead, J.D., and Hergt, J.M., 2001, Strontium, Neodymium and Lead Isotope Analyses of NIST Glass Certified Reference Materials: SRM 610, 612, 614: Geostandards and Geoanalytical Research, v. 25, p. 261–266, doi:10.1111/j.1751-908X.2001.tb00601.x.
```





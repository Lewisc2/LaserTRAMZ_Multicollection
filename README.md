# LaserTRAMZ LA-MC-ICP-MS
____________
## Citing and Related Documentation
#### See citation.cff file for citing information
____________
## Purpose
#### Open-source software built to reduce and handle time-resolved U-Pb data collected on a Nu multicollector ICP-MS. Two primary components are in this software. The first reduces the raw data. The second takes the reduced data and calculates ages.
____________
## Reference flowchart for repeat users that don't necessarily need to read the details below
[LaserTRAMZ_Reduction_Flowchart.pdf](https://github.com/user-attachments/files/19507647/LaserTRAMZ_Reduction_Flowchart.pdf)
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

____________
## Part Two: Age Reduction
#### Open the output files from part one and delete the first column and first row. Run the part two script through the terminal or an IDE. Copy and paste the file path into the appropriate input in the GUI tools. You should get a pop up window that looks like:
<img width="688" alt="Screenshot 2025-03-20 at 3 30 14 PM" src="https://github.com/user-attachments/assets/91c25d7b-d948-47cf-82d1-ab6a7c4a5c83" />

#### Input Options are in the order that follows:
* Primary Standard: Start typing the standard you wish to use, which should have a name the same as the table above. A dropdown will pop-up and you can either click the standard name or finish typing.
* Secondary Standard Excess Errors: Choose the standard you wish to use to evaluate excess variance according to the normalized ratios, c.a. Horstwood et al. (2016)
* List of buttons with sample names: Check those you want to use as validation standards. You need at least one for the program to run
* By Age, etc: Pb-Pb mass bias correction standard. If by Age, uses primary ZRM. Currently uses exponential correction of Woodhead (2002).
* Pb-Pb ratios list: Which measured Pb-Pb ratio to use for correcting Pb-Pb mass bias
* Primary 238U/235U, ...: Which standard to use for correcting the 238/235 mass bias. Primary 238U/235U uses primary ZRM and assumes 137.818 (Hiess et al., 2012)
* 238U/235U: Which ratio to use for correcting 38/35 bias. Currently only allows 38/35
* By Age, None: Drift correct according to reduced standard ages or not at all.
* Nearest Number: Nearest number of standards to use for correcting data. I.e., this is the sliding window of Gehrels et al. (2008)
* Decay Series Corrections: Whether to correct just for Common Pb or to correct for Common Pb and Th Diseuqilibrium
* Estimate Zircon [U/Th] By: How to estimate the [U/Th] concentration in zircon. For primary and selected secondary (same one as that used for excess variance), simply gets a factor according to the 238U and 232Th signals. Selecting Fitted regression standard will find all standards that you've selected (primary + those in the list of buttons) that have accepted concentrations and a simple linear regression will be used to make a calibration curve. If you have estimated the [U/Th] by some other way (e.g., split stream) you MUST manually enter these into the output from part one yourself. The column headers MUST be the following: [U] µg/g, [Th] µg/g, [U/Th].
* Calculate D [Th/U] From: First option uses inputs below to estimate D [Th/U] for the Th-disequilibrium correction (e.g., if you want to assume a constant value). Second option uses the estimated [U/Th] and the input melt [U/Th] in the input below.
* Th/U Zircon: Th/U ratio to use in zircon for Th disequilibrium correction
* Th/U Magma: Th/U ratio to use for host melt for Th disequilbrium correction
* Calculate Excess Variance From: First option increases 6/38 and 7/6 uncertainty equally until selected secondary ages have an MSWD of one. Second option increases 6/38 and 7/6 uncertainty until the normalized ratios of the preferred secondary standard (selected above) have an MSWD of one. Third option increases the uncertainty of the raw primary standard ratios until an MSWD of one is achieved.
* Exported Data Format: Simple is your best bet and has all the data you need to report. Annotated has additional output columns to help users assess sources of uncertainties. Full output is admittedly a bit of a mess and is mostly there for program development.
* Accept Reduction Parameters: Click this value to accept all your inputs
#### Once the parameters are accepted, input a sample name in the Text Sample Selector slot in the Gui tools. You should get a screen that looks like:
<img width="943" alt="Screenshot 2025-03-20 at 3 31 46 PM" src="https://github.com/user-attachments/assets/31ee88b1-1728-4c63-8166-3a7a08205f76" />

* The Concordia plot can be toggled from Tera-Waserburg to Wetherhill. By default the Stacey-Kramers model is used for the common Pb correction and the resultant ages are those projected onto Concordia from Common Pb through the data point. Common Pb ratios can be changed in the input on the left, as can the Th-disequilibrium parameters.
* The drift plot currently has some bugs and is mostly useful to load data first without drift correcting in order to assess how severe the drift is.
* The excess variance plot shows the ages (for secondary standard options) and ratios of standard data used to calculate excess variance. Black bars are 2s. Red is the additional excess variance.

____________
## Age and Error Calculation Additional Notes
#### Age calculations largely follow prior authors (Kosler and Sylvester, 2003; Pullen et al., 2018; Horstwood et al. (2016)). Decay constants are from Jaffey et al. (1971) (238U / 235U), Cheng et al. (2000) (230Th), and Le Roux and Glendenin (1963) (232Th) following the recommendations of Steiger and Jager (1977). Errors on decay constants are from those authors or, for U, include the uncertainty for counting statistics (238U: 0.16%, 235U: 0.21%; Mattinson (1987)) Data points are plotted on Tera-Wasserburg concordia and projected down from common Pb. If you have no common Pb input then the Stacey-Kramers model is assumed. Error propagation largely follows the community accepted protocol (Horstwood et al., 2016); although systematic uncertainty from the ID-TIMS determinations are included. There is also no full external error, as we don't know what the uncertainty for standard values is for your lab. All necessary error contributions are in the output file (see below). 
#### Mass bias calculations use the exponential fractionation law. Pb-Pb ratios in NIST are from Woodhead and Hergt (2001). 238U/235U ratios in NIST are from Duffin et al. (2015).
#### The following standards have published concentrations for U and Th. Note Plesovice is not included due to the known heterogeneity (Slama et al., 2008). Choosing a standard(s) that does not have published concentrations and selecting an option that includes that (those) standard(s) to estimate concentrations will result in erroneous concentration estimations for unknowns (values set to 1e-7 in the script to avoid divide by zero errors).
* Temora
* Fish Canyon
* R33
* 91500
* FC1
* Oracle
* OG1
____________
## Final File Output Information
#### Outputting the data will drop a file in the LaserTRAMZ folder. Output information is fairly routine for the 'Simple' output format. Errors incldue all sources of error. More detail below.
* measurement index: Global measurement number
* SampleLabel: Input sample label and corresponding sample measurement number
* t start - b end: ablation start time, ablation end time, time-zero intercept time, background start time, background end time
* 238U - 204Hg_1SE: Background and interference corrected mean intensities and 1 standard error on all masses. bdl if bdl
* U µg/g - f206: Concentration data and fraction of common Pb
* 206Pb/238U - SE% 207/204: Background and interference corrected ratios and 1 standard error in percent. Ratios with a c are the fully correctred ratios.
* Ratios followed by 'Age' are the ages. Full errors are labeled appropriately.
____________
## Installation and Use
#### We recommend running the program through [Anaconda](https://www.anaconda.com/download). You may also need to download [Git](https://github.com/git-guides/install-git).  After downloading, one may clone the repository by opening their terminal or Anconda prompt and running the following lines (one by one). It is best to create a virtual environment, which is included in the code block below:
```
git clone https://github.com/Lewisc2/LaserTRAMZ_Multicollection.git
cd /path/to/LaserTRAMZ
conda create -n LaserTRAMZ python==3.9.18
pip install -r requirements.txt
```
#### where /path/to/LaserTRAMZ is the file path to the cloned repository. You will need to accept the install by typing y then pressing enter when prompted. Once this is complete, the program can be run by opening Spyder from the Anaconda navigator and running the scripts, or,
```
cd /path/to/LaserTRAMZ_Multicollection
conda activate LaserTRAMZ_Multicollection
python LaserTRAMZ_MC_UPb_analytes.py
```
#### for part one and 
```
cd /path/to/LaserTRAMZ
conda activate LaserTRAMZ
python LaserTRAMZ_Mc_UPb_Concordia.py
```
#### for part two
#### To shut down the virtual environemnt, run the following
```
conda deactivate LaserTRAMZ_Multicollection
```

#### Demo videos
https://www.youtube.com/@charlestylerlewis9472/videos

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





# Installation
MicEMD(Modeling, Inversion and Classification in ElectroMagnetic Detection)

Prerequisite: Installing Python and Conda

Currently, MicEMD will run on Python 3.5, 3.6, 3.7 and 3.8. And MicEMD depends on some packages in Conda, so the Anaconda or Miniconda should be installed in your system.

Installing Packages

`conda install SimPEG --channel conda-forge`

`conda install pyqt`

`conda install scikit-learn=0.24.2`

Run MicEMD
Then **run mainwindow.py in your python IDE or in command line:**

`python mainwindow.py`

# GUI
![image](https://github.com/UndergroundDetection/MICEMD/blob/master/doc/image/GUI.png)
The left of the GUI are two tabs about parameters setting: FDEM detection and TDEM detection. We demonstrate the inversion application based on the FDEM detection and demonstrate the classification application based on the TDEM detection. From top to bottom on the right of the GUI are the tab-based results display, the output of the current program, and function buttons.

# A list of API methods
![image](https://github.com/UndergroundDetection/MICEMD/blob/master/doc/image/API.jpg)<br>

For more detailes, visit https://antsesame.github.io/

# The Simulation steps
The simulation can be built by following steps:
1. Firstly, create the target, detector, collection class of the forward modeling, and call the simulate interface to generate the observed data.
2. Then, for the inversion problem, set the inversion parameters and call the inverse interface to estimate the properties of the metal target; for the classification problem, call the preprocess interface to reduce the dimension of the observed data and call the classify interface to classify underground metal targets.
3. Finally, create the Handler class to analyze, show, and save the results.

Get the simulation examples, you can visit https://antsesame.github.io/.

# Installation tutorial
We provide a more detailed installation tutorial and document possible problems that may occur during the installation process. For more detailes, visit [Installation tutorial.md](https://github.com/UndergroundDetection/MICEMD/blob/master/Installation%20tutorial.md).

# Underground metal target detection
We provide a documentation to better explain the principles and processes of underground metal target detection, to make the use of the software more convenient. For more detailes, visit [Detection of underground metal targets.md](https://github.com/UndergroundDetection/MICEMD/blob/master/Detection%20of%20underground%20metal%20targets.md).

# Installation tutorial

## Prerequisite: Installing Python and Conda

### Anaconda <https://docs.anaconda.com/free/anaconda/install/windows/>

Please use the latest version of Python 3. For more information on the transition of the Python ecosystem to Python 3, please see the [Python 3 Statement](https://python3statement.org/). We highly recommend installing it using Anaconda and virtual environment.

1.  Download the [Anaconda installer](https://www.anaconda.com/download).

2.  (Optional) Anaconda recommends verifying the integrity of the installer after downloading it.

    [How do I verify my installer’s integrity?](https://docs.anaconda.com/free/anaconda/install/windows/ "How do I verify my installer’s integrity?")

3.  Go to your Downloads folder and double-click the installer to launch. To prevent permission errors, do not launch the installer from the [Favorites folder](https://docs.anaconda.com/free/troubleshooting/#distro-troubleshooting-favorites-folder).

4.  Click **Next**.

5.  Read the licensing terms and click **I Agree**.

6.  It is recommended that you install for **Just Me**, which will install Anaconda Distribution to just the current user account. Only select an install for **All Users** if you need to install for all users’ accounts on the computer (which requires Windows Administrator privileges).

7.  Click **Next**.

8.  Select a destination folder to install Anaconda and click **Next**. Install Anaconda to a directory path that does not contain spaces or unicode characters. For more information on destination folders, see the [FAQ](https://docs.anaconda.com/free/working-with-conda/reference/faq/#distribution-faq-windows-folder).

    **Caution**：Do not install as Administrator unless admin privileges are required.

    [![](https://docs.anaconda.com/_images/win-install-destination1.png)](https://docs.anaconda.com/_images/win-install-destination1.png)

9.  Choose whether to add Anaconda to your PATH environment variable or register Anaconda as your default Python. We **don’t recommend** adding Anaconda to your PATH environment variable, since this can interfere with other software. Unless you plan on installing and running multiple versions of Anaconda or multiple versions of Python, accept the default and leave this box checked. Instead, use Anaconda software by opening Anaconda Navigator or the Anaconda Prompt from the Start Menu.

    [![](https://docs.anaconda.com/_images/win-install-options1.png)](https://docs.anaconda.com/_images/win-install-options1.png)

10. Click **Install**. If you want to watch the packages Anaconda is installing, click Show Details.

11. Click **Next**.

12. Optional: To learn more about Anaconda’s cloud notebook service, go to <https://www.anaconda.com/code-in-the-cloud>.

    [![](https://docs.anaconda.com/_images/win-install-cloud-notebook1.png)](https://docs.anaconda.com/_images/win-install-cloud-notebook1.png)

    Or click **Continue** to proceed.

13. After a successful installation you will see the “Thanks for installing Anaconda” dialog box:

    [![](https://docs.anaconda.com/_images/win-install-complete1.png)](https://docs.anaconda.com/_images/win-install-complete1.png)

14. If you wish to read more about Anaconda.org and how to get started with Anaconda, check the boxes “Anaconda Distribution Tutorial” and “Learn more about Anaconda”. Click the **Finish** button.

15. [Verify your installation](https://docs.anaconda.com/free/anaconda/install/verify-install/).

    > [Installing on macOS](https://docs.anaconda.com/free/anaconda/install/mac-os/ "Installing on macOS")
    >
    > [Installing on Linux](https://docs.anaconda.com/free/anaconda/install/linux/ "Installing on Linux")

### Conda&#x20;

You should have already installed conda before beginning this getting started guide.&#x20;

Starting conda

Conda is available on Windows, macOS, or Linux and can be used with any terminal application (or shell).

*   \*\*Windows：\*\*Open either the Command Prompt (cmd.exe) or PowerShell.

*   \*\*macOS：\*\*Open Launchpad→Open the Other application folder.→Open the Terminal application.

*   \*\*Linux：\*\*Open a terminal window.

#### Creating environments

Conda allows you to create separate environments, each containing their own files, packages, and package dependencies. The contents of each environment do not interact with each other.

The most basic way to create a new environment is with the following command:

    conda create -n <env-name>

To add packages while creating an environment, specify them after the environment name:

    conda create -n myenvironment python numpy pandas

For more information on working with environments, see [Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

#### Listing environments

To see a list of all your environments:

    conda info --envs

A list of environments appears, similar to the following:

    conda environments:

       base           /home/username/Anaconda3
       myenvironment   * /home/username/Anaconda3/envs/myenvironment

Tip

The active environment is the one with an asterisk (\*).

To change your current environment back to the default `base`:

    conda activate

Tip

When the environment is deactivated, its name is no longer shown in your prompt, and the asterisk (\*) returns to `base`. To verify, you can repeat the `conda info --envs` command.

#### Specifying channels

Channels are locations (on your own computer or elsewhere on the Internet) where packages are stored. By default, conda searches for packages in its [default channels](https://conda.io/projects/conda/en/latest/user-guide/configuration/settings.html#default-channels).

If a package you want is located in another channel, such as conda-forge, you can manually specify the channel when installing the package:

    conda install conda-forge::numpy

You can also override the default channels in your .condarc file. For a direct example, see [Channel locations (channels)](https://conda.io/projects/conda/en/latest/user-guide/configuration/settings.html#config-channels) or read the entire [Using the .condarc conda configuration file](https://conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html).

Tip

Find more packages and channels by searching [Anaconda.org](https://www.anaconda.org/).

#### Updating conda

To see your conda version, use the following command:

    conda --version

No matter which environment you run this command in, conda displays its current version:

    conda 23.10.0

Note

If you get an error message `command not found: conda`, close and reopen your terminal window and verify that you are logged into the same user account that you used to install conda.

To update conda to the latest version:

    conda update conda

Conda compares your version to the latest available version and then displays what is available to install.

Tip

We recommend that you always keep conda updated to the latest version. For conda's official version support policy, see [CEP 10](https://github.com/conda-incubator/ceps/blob/main/cep-10.md).

## Installing packages

### Single package installation

You can also install packages into a previously created environment. To do this, you can either activate the environment you want to modify or specify the environment name on the command line:

    # via environment activation
    conda activate myenvironment

    conda install SimPEG --channel conda-forge
    conda install pyqt
    conda install scikit-learn=0.24.2

    # via command line option
    conda install --name myenvironment matplotlib

For more information on searching for and installing packages, see [Managing packages](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html).

**Note**: When you cannot install using `conda install`, you can consider searching for the package name to be installed from [http://Anaconda.org](http://anaconda.org/), obtain the command to install the package, and install it.

### Install third-party library&#x20;

#### freeze

`requirements.txt` is used to record all dependent packages and version numbers of the project. It only requires a simple pip command to complete.

    pip freeze > requirements.txt

> The requirement.txt file is output on the desktop by default.
>
> Note: View the source file. The freeze command of pip is used to generate the pip class library list of the current project into the requirements.txt file.

Then

    pip install -r requirements.txt

It is convenient to install all the dependency packages in `requirements.txt` at once.

The `requirements.txt` file looks like this:

    Django=1.3.1
    South>=0.7
    django-debug-toolbar

***

#### pipreqs

The advantage of pipreqs is that it can automatically discover which class libraries are used by scanning the project directory and automatically generate a dependency list. The disadvantage is that there may be some deviations, which need to be checked and adjusted by yourself.

Installation

    pip install pipreqs

Switch to the project directory and use:

    pipreqs ./

If it is a Windows system, a coding error (UnicodeDecodeError: 'gbk' codec can't decode byte 0xa8 in position 24: illegal multibyte sequence) will be reported. This is caused by encoding problems. Just add the encoding parameter, as follows:

    pipreqs ./ --encoding=utf-8

After generating the `requirements.txt` file,  all dependencies can be downloaded based on this file.

    pip install -r requriements.txt

<!---->

    Usage:

    pipreqs [options] <path>

    Options:
         --use-local only use local package information instead of querying PyPI
         --pypi-server <url>Use custom PyPi server
         --proxy <url>Use Proxy, parameters will be passed to the request library. You can also set
        
         Environment parameters in the terminal:
         $export HTTP_PROXY="http://10.10.1.10:3128"
         $export HTTPS_PROXY="https://10.10.1.10:1080"
         --debug prints debugging information
         --ignore <dirs> ...ignore extra directories
         --encoding <charset>Open file using encoding parameters
         --savepath <file>Save the list of requirements in the given file
         --print output list of requirements to standard output
         --force overwrite existing requirements.txt
         --diff <file>Compare modules in requirements.txt with project imports.
         --clean <file>Clean requirements.txt by removing modules not imported in the project.

## Notes during installation

Due to various factors such as the configuration environment and the version of packages, you may encounter some problems when using this software. Here are some for reference only:

    Traceback (most recent call last):
    File ”articleexam1.py”, line 2, in <module> 
    import  MicEMD.fdem as fdem
    File ”/home/<user>/MICEMD/src/MicEMD/init.py”,line1, in <module> 
    from .optimization  import *
    File ”/home/<user>/MICEMD/src/MicEMD/optimization/init.py”, line1, in
     <module> 
     from ._numopt  import *
    File ”/home/<user>/MICEMD/src/MicEMD/optimization/numopt.py”, line14, in
     <module> 
     from scipy.optimize.optimize import _linesearchwolfe12, _LineSearchError
    ImportError: cannot import name ’ _linesearchwolfe12’ from 
    ’scipy.optimize.optimize’
    (/home/<user>/anaconda3/lib/python3.8/site=packages/scipy/optimize/optimize.py)

It can be solved by modifying in the file MICEMD/src/MicEMD/optimization/ numopt.py the **from scipy.optimize.optimize import \_linesearchwolfe12, \_LineSearchError** to **from scipy.optimize import \***

***

    Traceback (most recent call last):
    File ”articleexam2.py”, line54, in <module> 
    handler.save_cls_res(cls_material_res, ’cls_material_res.csv’)
    File ”/home/<user>/MICEMD/src/MicEMD/handler/handler.py”, line 1254, in
    save_cls_res
    y_true=np.array(cls_res[’y_true’], dtype=np.int)
    File ”/home/<user>/anaconda3/lib/python3.8/site-packages/numpy/_ _init_ _.py”, 
    line305, in _ _getattr_ _
    raise AttributeError(_ _former_ _attrs[attr])
    AttributeError: module ’numpy’ has no attribute ’int’.

numpy.int was deprecated in NumPy 1.20 and removed in NumPy 1.24. Therefore, using numpy.int in versions after 1.24 will raise an error. Using Python’s int type is the easiest replacement.&#x20;

It can be solved this by substituting **np.int** with **int** in the file MICEMD/src/MicEMD/handler/handler.py

***

    Traceback (most recent call last):
    File ”articleexam2.py”, line54, in <module> 
    handler.save_cls_res(cls_material_res, ’cls_material_res.csv’)
    File ”/home/<user>/MICEMD/src/MicEMD/handler/handler.py”, line1261, in
    save_cls_res
    accuracy=np.zeros(shape=(y_true.shape[0], 1), dtype=np.str)
    File ”/home/<user>/anaconda3/lib/python3.8/site=packages/numpy/init.py”, 
    line 305, in _ _getattr_ _
    raise AttributeError(_ _former_attrs_ _[attr])
    AttributeError: module ’numpy’ has no attribute ’str’.

numpy.scr was deprecated in NumPy 1.20. It can be solved this by substituting **np.str** with **str** in the file MICEMD/src/MicEMD/handler/handler.py

***

    Traceback (most recent call last):
    File ”mainwindow.py”, line 567, in <module> 
    m=MainWindow()
    File ”mainwindow.py”, line 88, in _ _init_ _
    self.get_fdem_simulation_parameters()
    File ”mainwindow.py”, line 283, in get_fdem_simulation_parameters
    show_fdem_detection_scenario(self.fig_scenario, self.ftarget, 
    self.fcollection)
    File ”/home/<user>/MICEMD/src/utilities/show.py”, line57, in
    show_fdem_detection_scenario
    ax=fig.gca(projection=’3d’)
    TypeError: gca() got an unexpected Keyword argument ’projection’

This error happens due to Matplotlib update. It can be solved by modifying in the file MICEMD/src/utilities/show\.py the line **ax \= fig.gca(projection\=’3d’)** in **ax \= fig.add subplot(projection\=’3d’)**

***

    Traceback (most recent call last):
    File "/home/user/MICEMD/src/utilities/threadSet.py", line 57, in run
    forward_result = (f.simulate(self.target, self.detector, self.collection, 'simpeg'),)
    File "/home/user/MICEMD/src/MicEMD/fdem/simulation.py", line 91, in simulate
    result = simulation.pred()
    File "/home/user/MICEMD/src/MicEMD/fdem/simulation.py", line 61, in pred
    result = self.model.dpred()
    File "/home/user/MICEMD/src/MicEMD/fdem/model.py", line 140, in dpred
    simulation = fdem.simulation.Simulation3DMagneticFluxDensity(
    File "/home/user/anaconda3/lib/python3.8/site-packages/properties/base/base.py", line 280, in _ _call_ _
    obj. _ _init_ _ (*args, **kwargs)
    File "/home/user/anaconda3/lib/python3.8/site-packages/SimPEG/electromagnetics/frequency_domain/simulation.py", line 389, in _ _init _ _
    super(Simulation3DMagneticFluxDensity, self). _ _init _ _ (mesh, **kwargs)
    File "/home/user/anaconda3/lib/python3.8/site-packages/SimPEG/base/pdesimulation.py", line 439, in _ _init _ _
    super().init(mesh, **kwargs)
    File "/home/user/anaconda3/lib/python3.8/site-packages/SimPEG/base/pdesimulation.py", line 491, in _ _init _ _
    super()._ _init _ _ (mesh, **kwargs)
    File "/home/user/anaconda3/lib/python3.8/site-packages/SimPEG/simulation.py", line 215, in _ _init _ _
    super(BaseSimulation, self). _ _init _ _ (**kwargs)
    File "/home/user/anaconda3/lib/python3.8/site-packages/properties/base/base.py", line 314, in _ _init _ _
    raise AttributeError(
    AttributeError: Keyword input 'Solver' is not a known property or attribute of Simulation3DMagneticFluxDensity.
    Aborted

The proposed solution is just a workaround. Probably better is to change imports in  `.src/MicEMD/fdem/model.py` to :

    try:
        from pymatsolver import Pardiso as Solver
    except ImportError:
        from .utils.solver_utils import SolverLU as Solver

instead of

    try:
        from pymatsolver import Pardiso as Solver
    except ImportError:
        from SimPEG import SolverLU as Solver


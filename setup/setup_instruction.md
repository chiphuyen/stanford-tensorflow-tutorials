Tensorflow supports both Python 2.7 and Python 3.3+. <b>Note that for Windows, TensorFlow supports only 64-bit Python 3.5.</b>
For this course, I will use Python 2.7. But you’re welcome to use either Python 2 or Python 3 for the assignments. The starter code, though, will be in Python 2.7

Google has a pretty detailed instruction on how to download and setup Tensorflow. You can follow it here: https://www.tensorflow.org/get_started/os_setup

Unless your computer has GPU, you should install Tensorflow without GPU support. My recommendation is always set up Tensorflow using virtualenv. For the list of dependencies, please consult the file requirements.txt. This list will be updated as the course progresses.

Below is a simpler instruction on how to install tensorflow for people using Mac OS. If you have any problem installing Tensorflow, feel free to post it on Piazza: piazza.com/stanford/winter2017/cs20si

## Install TensorFlow<br>
### For Mac OS

If you get “permission denied” error in any command, use “sudo” in front of that command.

You will need pip (or pip3 if you use Python 3), and virtualenv.

Step 1: set up pip and virtual environment
```bash
$ sudo easy_install pip 
$ sudo easy_install --upgrade six
$ pip install virtualenv
```

Step 2: set up a project directory. You will do all work for this class in this directory
```bash
$ mkdir [my project]
```

Step 3: set up virtual environment for the project directory. 
```bash
$ cd [my project]
$ virtualenv venv --distribute
```
These commands create a venv subdirectory in your project where everything is installed.

Step 4: to activate the virtual environment 
```bash
$ source venv/bin/activate
```

If you type:
```bash
$ pip freeze
```

You will see that nothing is shown, which means no package is installed in your virtual environment. So you have to install all packages that you need. For the list of packages you need for this class, refer to requirements.txt
Step 5: Install Tensorflow and other dependencies
```bash
$ pip install tensorflow
$ pip freeze > requirements.txt
```

Step n: 
To exit the virtual environment, use:
```bash
$ deactivate
```

If you want your virtual environment to inherit globally installed packages, (not recommended), use:
```bash
$ virtualenv venv --distribute --system-site-packages
```
### For Ubuntu


### For Windows


### On the cloud
If you don't want to install TensorFlow, you can use TensorFlow over the web.

#### SageMath
You can use Tensorflow over the web at https://cloud.sagemath.com/
Simply click on the link, create an account (or log in with your GitHub), and create a TensorFlow project.

#### Jupyter
You can also use Jupyter notebook to write TensorFlow programs.

# Possible set up problems
## Matplotlib
If you have problem with using Matplotlib in virtual environment, here is a simple fix. <br>
If you installed matplotlib using pip, there is a directory in you root called ~/.matplotlib.
Go there and create a file ~/.matplotlib/matplotlibrc there and add the following code: ```backend: TkAgg```

Or you can simply add this after importing matplotlib: ```matplotlib.use("TkAgg")```

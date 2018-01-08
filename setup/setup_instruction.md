Please follow the official instruction to install TensorFlow [here](https://www.tensorflow.org/install/). For this course, I will use Python 3.6 and TensorFlow 1.4. You’re welcome to use either Python 2 or Python 3 for the assignments. The starter code, though, will be in Python 3.6. You don't need GPU for most code examples in this course, though having GPU won't hurt. If you install TensorFlow on your local machine, my ecommendation is always set up Tensorflow using virtualenv. 

For the list of dependencies, please consult the file requirements.txt. This list will be updated as the course progresses. 

There are a few things to note:
- As of version 1.2, TensorFlow no longer provides GPU support on macOS.
- On macOS, Python 3.6 might gives warning but still works.
- TensorFlow with GPU support will only work with CUDA® Toolkit 8.0 and cuDNN v6.0, not the newest CUDA and cnDNN version. Make sure that you install the correct CUDA and cuDNN versions to avoid frustrating issues.
- On Windows, TensorFlow supports only 64-bit Python 3.5 anx Python 3.6.
- If you see the warning:
```bash
Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
```
it's because you didn't install TensorFlow from sources to take advantage of all these settings. You can choose to install TensorFlow from sources -- the process might take up to 30 minutes. To silence the warning, add this before importing TensorFlow: <br>

```bash
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
```

- If you want to install TensorFlow from sources, keep in mind that TensorFlow doesn't officially support building TensorFlow on Windows. On Windows, you may try using the highly experimental Bazel on Windows or TensorFlow CMake build.

Below is a simpler instruction on how to install TensorFlow on macOS. If you have any problem installing Tensorflow, feel free to post it on [Piazza](piazza.com/stanford/winter2018/cs20)

If you get “permission denied” error in any command, use “sudo” in front of that command.

You will need pip3 (or pip if you use Python2), and virtualenv.

Step 1: install python3 and pip3. Skip this step if you already have both. You can find the official instruction [here](http://docs.python-guide.org/en/latest/starting/install3/osx/)

Step 2: upgrade six
```bash
$ sudo easy_install --upgrade six
```

Step 3: install virtualenv. Skip this step if you already have virtualenv
```bash
$ pip3 install virtualenv
```

Step 4: set up a project directory. You will do all work for this class in this directory
```bash
$ mkdir cs20
```

Step 5: set up virtual environment with python3
```bash
$ cd cs20
$ python3 -m venv .env
```
These commands create a venv subdirectory in your project where everything is installed.

Step 6: activate the virtual environment 
```bash
$ source .env/bin/activate
```

If you type:
```bash
$ pip3 freeze
```

You will see that nothing is shown, which means no package is installed in your virtual environment. So you have to install all packages that you need. For the list of packages you need for this class, you can see/download the list of requirements in [the setup folder of this repository](https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/setup/requirements.txt).

Step 7: Install Tensorflow and other dependencies
```bash
$ pip3 install -r requirements.txt
```

Step n: 
To exit the virtual environment, use:
```bash
$ deactivate
```

### Other options
#### Floydhub
Floydhub has a clean, GitHub-like interface that allows you to create and run TensorFlow projects.

# Possible set up problems
## Matplotlib
If you have problem with using Matplotlib in virtual environment, here are two simple ways to fix. <br>
1. If you installed matplotlib using pip, there is a directory in you root called ~/.matplotlib.
Go there and create a file ~/.matplotlib/matplotlibrc there and add the following code: ```backend: TkAgg```
2. After importing matplotlib, simply add: ```matplotlib.use("TkAgg")```

If you run into more problems, feel free to post your questions on [Piazza](https://piazza.com/stanford/winter2018/cs20) or email us cs20-win1718-staff@lists.stanford.edu.
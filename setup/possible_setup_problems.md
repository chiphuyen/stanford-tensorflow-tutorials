## Matplotlib
If you have problem with using Matplotlib in virtual environment, here is a simple fix. <br>
If you installed matplotlib using pip, there is a directory in you root called ~/.matplotlib.
Go there and create a file ~/.matplotlib/matplotlibrc there and add the following code: ```backend: TkAgg```

Or you can simply add this after importing matplotlib: ```matplotlib.use("TkAgg")```

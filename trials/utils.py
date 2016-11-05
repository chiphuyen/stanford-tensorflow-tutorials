import numpy as np
import pylab
from matplotlib.backends.backend_pdf import PdfPages

'''
    Supposed there are n arguments
    The first parameter should be for the x axis
    The second parameter should be the label of the x axis

    The last parameter should be the label of the y axis
'''
def graph(*args):
    assert len(args) >= 4, "graph requires at least 4 parameters"

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    n = len(colors)

    X = args[0]
    pylab.xlabel(args[1])
    pylab.ylabel(args[-1])

    for i in range(2, len(args)-1, 2):
        print "i", i
        pylab.plot(X, args[i], colors[i%n], label=args[i+1])
        pylab.legend(loc='lower left')
    pp = PdfPages("graphs/" + args[1] + "_" + args[-1] + ".pdf")
    pylab.savefig(pp, format="pdf")
    pp.close()
    pylab.show()
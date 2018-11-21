# moments
Python tools for moments data.

*Note:* 
* this is only for moments data output from PPMstar version 1 (PPMstar1)
* access to moments data from PPMstar2 is via the PyPPM module `ppm.py`

Moments is a python package for working with PPM FVandMoms48 data.

Examples
--------
```
import moments
from matplotlib import pyplot

data = moments.Moments("/npod2/users/lsiemens/PPM/D15")
print(data.fields)#print list of fields
print(data.cycles)#print list of cycles
Rho = data.get("Rho", 175)

pyplot.plot(data.raxis, moments.radprof(Rho))
pyplot.show()
```

Testing
-------
There are some unit test in ./moments/tests. To run the tests use the command
```
    python setup.py test
```

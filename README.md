# moments
python tools for moments data

moments is a python package for working with PPM FVandMoms48 data.

examples
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

testing
-------
There are some unit test in ./moments/tests. To run the tests use the command
```
    python setup.py test
```

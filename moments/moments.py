#################################################################################
#                                                                               #
# moments.py - Tools for accessing and visualising PPM moments data. Depends on #
#              the nugridpy package developed by the NuGrid collaboration.      #
#                                                                               #
# (c) 2016 Luke Siemens                                                         #
#                                                                               #
#################################################################################

"""
moments.py

moments is a Python module for reading FVandMoms48-xxxx.bobxxx files. The command
line tool read_ppm is used to decompress and unscramble the files and the command
line tool e3d is used to compute derived quantities from the raw data.

Example
-------
>>> from matplotlib import pyplot
>>> import moments
>>> data = moments.Moments() #assuming there is FVandMoms data in the CWD
>>> print(data.fields)
["FV", "DivUav", "Vortmag", "Prs", ... , "rcoord"]
>>> print(data.cycles)
["0001", "0002", ... , "0209", "0210"]
>>> cyc = 175
>>> rho = data.get("Rho", cyc)
>>> rhoux = data.get("RhoUx", cyc)
>>> rhouy = data.get("RhoUy", cyc)
>>> rhouz = data.get("RhoUz", cyc)
>>> rhoU = moments.array([rhoux, rhouy, rhouz])
>>> U = rhoU/rho
>>> magU = moments.norm(U)
>>>
>>> # plot 2d cross section
>>> pyplot.imshow(magU[:, :, int(magU.shape[0]/2.0)])
>>> pyplot.show()
>>>
>>> # plot radial profile
>>> fv = data.get("FV", 300, nearest=True)
>>> pyplot.plot(data.raxis, moments.radprof(fv))
>>> pyplot.show()
"""

__all__ = ["Moments", "set_moments_path"]

#from data_plot import DataPlot
import numpy
import os
import copy

from . import core

#initalize _PATH
_PATH = None

def set_moments_path(path=None):
    """ 
    Set the path to be used as the default path when looking for PPM run directories
    or PPM FVamdMoms48 data.
    
    Parameters
    ----------
    path : string, optional
        Path to be used as the default path. If path is None then relative paths
        are assumed to be relitive to the current working directory. The default
        is None.
    """
    
    global _PATH
    if path is not None:
        _PATH = os.path.abspath(path) + "/"
    else:
        _PATH = None

class Moments:
    """ 
    Moments is a data structure for loading and accessing PPM moments data.
    """

    _current_id = 0
    _global_path = None
    
    def __init__(self, sldir="./", cache_dir="./cache", use_e3d=True):
        """ 
        Parameters
        ----------
        sldir : string, optional
            The path to a PPM run directory, or to PPM FVandMoms48 data. The
            default is "./".
        cache_dir : string, optional
            Path to directory for read_ppm and e3d temparary and cache files.
            The default is "./cache".
        use_e3d : boolean, optional
            Toggle whether the e3d command line tool should be used to compute
            derived quantities. The default is True.
        """
        
        global _PATH
        
        self._current_id += 1
        
        if _PATH is not None:
            if os.path.isdir(_PATH + sldir):
                sldir = _PATH + sldir
    
        self._use_e3d = use_e3d
        self._wrapper = core.Wrapper(cache_dir)
        self._ppmdir = core.ppmdir.get_ppmdir(sldir, all_files=False)
        
        self._id = "dumpfile-" + str(self._current_id) + "-000"
        
        if "read_ppm" not in self._ppmdir.get_ppminfile_types():
            # Todo add waring
            print("Warning no read_ppm.in files found in directory tree: " + str(sldir))

        if "FVandMoms48" not in self._ppmdir.get_bobfile_types():
            # Todo add waring
            print("Warning no FVandMoms files found in directory tree: " + str(sldir))
    
    def set_tool_paths(self, dir_read_ppm=None, dir_e3d=None):
        """
        Set path to command line tools e3d and read_ppm. Set the path to None
        if they are in the system path.

        Parameters
        ----------
        dir_read_ppm : string, optional
            The path to read_ppm, set to None if read_ppm is in the system Path.
            The default is None.
        dir_e3d : string, optional
            The path to e3d, set to None if read_ppm is in the system Path.  The
            default is None.
        """
                                                                                                                    
        self._wrapper.set_tool_paths(dir_read_ppm, dir_e3d)
        
    def load_read_ppm(self, fname):
        """ 
        Load external read_ppm.in file.
        
        Parameters
        ----------
        fname : string
            Name of read_ppm.in file.
        """
        dir, fname = os.path.split(fname)
        self._ppmdir._load_ppminfiles(dir, [fname])

        if "read_ppm" not in self._ppmdir.get_ppminfile_types():
            # Todo add waring
            print("Warning no read_ppm.in files found in directory tree: " + str(sldir))
        
    def getCycles(self):
        """ 
        Get list of avaliable cycles.
        
        Returns
        -------
        cycles : list
            A list of cycle numbers
        """
        
        return self._ppmdir.get_dumps("FVandMoms48")

    def getFields(self):
        """ 
        Get list of avaliable data fields.
        
        Returns
        -------
        fields : list
            A list of data fields
        """
        
        return core.PPMField._fields + ["xcoord", "ycoord", "zcoord", "rcoord"]
        
    def getAxis(self, axis):
        """ 
        Get spesified axis.
        
        Parameters
        ----------
        axis : string
            Axis label, one of "x", "y", "z" or "r".
        
        Returns
        -------
        axis : numpy.ndarray
            The axis as a numpy array.
        """
        
        try:
            return self._get_axis(axis)
        except TypeError:
            self._wrapper.load_read_ppm_in(self._ppmdir.get_dumpfiles("read_ppm", 0, nearest=True)[0])
            return self._get_axis(axis)
    
    def get(self, attri, cycle, nearest=False):
        """ 
        Get data field for a given cycle.
        
        Parameters
        ----------
        attri : string
            Name of field to get.
        cycle : string or integer
            Cycle / Dump number.
        
        Returns
        -------
        field : numpy.ndarray or moments.core.PPMField
            If this class was initalized with use_e3d set to True then the data
            field will be returned as a moments.core.PPMField instance, otherwise
            it will be returned as a numpy.ndarray instance.
        """
        
        if attri in ["xcoord", "ycoord", "zcoord", "rcoord"]:
            Rho = self.get("Rho", cycle, nearest=nearest)
            if attri == "xcoord":
                if self._use_e3d:
                    return Rho._coord("x")
                else:
                    shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)
                    field = numpy.empty(shape=shape)
                    field[:] = self.xaxis.reshape((shape[0], 1, 1))
                    return field
            elif attri == "ycoord":
                if self._use_e3d:
                    return Rho._coord("y")
                else:
                    shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)
                    field = numpy.empty(shape=shape)
                    field[:] = self.yaxis.reshape((1, shape[1], 1))
                    return field
            elif attri == "zcoord":
                if self._use_e3d:
                    return Rho._coord("z")
                else:
                    shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)
                    field = numpy.empty(shape=shape)
                    field[:] = self.zaxis.reshape((1, 1, shape[2]))
                    return field
            elif attri == "rcoord":
                return core.ppmfield.norm(core.ppmfield.array([self.get("xcoord", cycle, nearest=nearest), self.get("ycoord", cycle, nearest=nearest), self.get("zcoord", cycle, nearest=nearest)]))

        bobfile = self._ppmdir.get_dumpfiles("FVandMoms48", cycle, nearest=nearest)[0]
        self._wrapper.load_read_ppm_in(self._ppmdir.get_dumpfiles("read_ppm", cycle, nearest=True)[0])
        if self._use_e3d:
            outfile = self._id
        else:
            outfile = "bofs:" + self._id[:-4]
        self._wrapper.update_read_ppm_in(file="\"" + bobfile + "\"", readvars="\"" + " ".join(core.PPMField._fields) + "\"", outfile=outfile)
            
        if self._use_e3d:
            shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)
            return core.PPMField(wrapper=copy.deepcopy(self._wrapper), statement=attri, shape=shape)
        else:
            self._wrapper.process_bobfiles()
            return self._wrapper.read_bof(self._id[:-3] + attri + ".bof")

    def fromradprof(self, raxis, radprof=None):
        """
        Generate field from radial profile.
        
        Parameters
        ----------
        raxis : array
            Radial axis for the radial profile. If radprof is None then raxis
            is taken to be the radial profile and the radial axis is assumed
            to be self.raxis.
        radprof : array, optional
            Radial profile. If None then raxis is assumed to be the radial
            profile. The default is None.
        """
        
        if radprof is None:
            radprof, raxis = raxis, self.raxis
        radprof, raxis = numpy.array(radprof), numpy.array(raxis)
        if len(raxis) != len(radprof):
            raise ValueError("raxis and radprof must have the same length.")
        
        shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)
        if None in shape:
            self._wrapper.load_read_ppm_in(self._ppmdir.get_dumpfiles("read_ppm", 0, nearest=True)[0])
            shape = (self._wrapper._xresolution, self._wrapper._yresolution, self._wrapper._zresolution)

        field = numpy.empty(shape)

        dr = (raxis[-1] - raxis[0])/float(len(raxis) - 1)
        drinv = 1.0/dr
        if drinv < 0:
            dr = -dr
            drinv = -drinv
            raxis = raxis[::-1]
            radprof = radprof[::-1]

        fnmax = float(len(raxis) - 2)
        nmomav = 4
        fac = 1.0/float(nmomav)
        dx = (self.xaxis[-1] - self.xaxis[0])/float(len(self.xaxis) - 1)
        dy = (self.yaxis[-1] - self.yaxis[0])/float(len(self.yaxis) - 1)
        dz = (self.zaxis[-1] - self.zaxis[0])/float(len(self.zaxis) - 1)
        dx1 = dx*fac
        dy1 = dy*fac
        dz1 = dz*fac
        off = 0.5*float(nmomav) - 0.5

        x0, y0, z0 = numpy.empty(shape=shape), numpy.empty(shape=shape), numpy.empty(shape=shape)
        x0[:] = self.xaxis.reshape((shape[0], 1, 1))
        y0[:] = self.yaxis.reshape((1, shape[1], 1))
        z0[:] = self.zaxis.reshape((1, 1, shape[2]))
        
        # iterate computation over x axis. During testing with a 1536 run when iteration was not used memorry
        # allocation errors occured and when only iteration was used the calculation was too slow ~1hour.
        # Should probably be implemented in C or Fortran in the future.
        for ix in range(len(x0)):
            X, Y, Z = numpy.indices((nmomav, nmomav, nmomav)) - off
            z = z0[ix, :, :, numpy.newaxis, numpy.newaxis, numpy.newaxis] + (dz1*(Z))[numpy.newaxis, numpy.newaxis, :, :, :]
            y = y0[ix, :, :, numpy.newaxis, numpy.newaxis, numpy.newaxis] + (dy1*(Y))[numpy.newaxis, numpy.newaxis, :, :, :]
            x = x0[ix, :, :, numpy.newaxis, numpy.newaxis, numpy.newaxis] + (dx1*(X))[numpy.newaxis, numpy.newaxis, :, :, :]
    
            frac, ir0 = numpy.modf(numpy.minimum(fnmax, drinv*numpy.sqrt(x**2 + y**2 + z**2)))
            ir0 = ir0.astype(dtype=numpy.int32)
            value = numpy.sum((1.0 - frac)*radprof[ir0] + frac*radprof[ir0 + 1], axis=(2, 3, 4))
            field[ix] = value
        return field*(fac**3)

    @property
    def cycles(self):
        """ 
        List of avaliable cycles derived from self.getCycles()
        """

        return self.getCycles()

    @property
    def fields(self):
        """ 
        List of avaliable data fields derived from self.getFields()
        """

        return self.getFields()

    @property
    def xaxis(self):
        """ 
        xaxis derived from self.getAxis()
        """

        return self.getAxis("x")

    @property
    def yaxis(self):
        """ 
        yaxis derived from self.getAxis()
        """

        return self.getAxis("y")

    @property
    def zaxis(self):
        """ 
        zaxis derived from self.getAxis()
        """

        return self.getAxis("z")

    @property
    def raxis(self):
        """ 
        raxis derived from self.getAxis()
        """

        return self.getAxis("r")

    def _get_axis(self, axis):
        """ 
        Get spesified axis.
        
        Parameters
        ----------
        axis : string
            Axis label, one of "x", "y", "z" or "r".
        
        Returns
        -------
        axis : numpy.ndarray
            The axis as a numpy array.
        """
        
        axis = axis.lower()
        if axis == "x":
            return numpy.linspace(-self._wrapper._xmax, self._wrapper._xmax, self._wrapper._xresolution)*(1.0-1.0/self._wrapper._xresolution)
        elif axis == "y":
            return numpy.linspace(-self._wrapper._ymax, self._wrapper._ymax, self._wrapper._yresolution)*(1.0-1.0/self._wrapper._yresolution)
        elif axis == "z":
            return numpy.linspace(-self._wrapper._zmax, self._wrapper._zmax, self._wrapper._zresolution)*(1.0-1.0/self._wrapper._zresolution)
        elif axis == "r":
            return numpy.linspace(self._wrapper._xmax/(self._wrapper._xresolution - 1.0), self._wrapper._xmax, int(self._wrapper._xresolution/2.0))*(1.0-1.0/self._wrapper._xresolution)

################################################################################
#                                                                              #
# ppmdir.py - A python module for accessing files in PPM run directories.      #
#                                                                              #
# (c) 2016 Luke Siemens                                                        #
#                                                                              #
################################################################################

"""
ppmdir.py

ppmdir is a Python module for analyzing the contents of a PPM run directory and
provides an interface for extracting file paths.

In this module the PPM data and configuration files are categorized into four types.

 - profile: includes YProfiles and RProfiles.
 - bobfile: includes FVandMoms48, FV-hires-01 ect.
 - ppminfile: includes read_ppm.in files and the associated .dat files.
 - hvfile: includes FV-hires-01, Lg10Vort-01 ect given they are in hv format.
"""

import shutil
import os

__all__ = ["get_ppmdir", "PPMDir"]

def get_ppmdir(dir="./", all_files=True):
    """ 
    Determine which directory format is being used and initaliaze an instance of
    the appropriate PPMDir subclass.
    
    Parameters
    ----------
    dir : string, optional
        Path to the target directory. The default is "./"
    all_files : bool, optional
        Toggles whether all files should be recorded or only recognized types
        and formats, if True all files will be recorded. The default is True

    Returns
    -------
    PPMDir subclass
         A initalized class instance which inherets from PPMDir.
    """
    
    dir = os.path.abspath(dir) + "/"
    contents = os.listdir(dir)
    dirs = [file for file in contents if os.path.isdir(dir + file)]
    files = [file for file in contents if os.path.isfile(dir + file)]

    if any([directory.isdigit() for directory in dirs]):
        return new_format(dir, all_files=all_files)
    elif any([any([directory.startswith(prefix) for prefix in ["FVandMoms", "HV", "RProfile", "YProfile"]]) for directory in dirs]):
        return compact_format(dir, all_files=all_files)
    else:
        return old_format(dir, all_files=all_files)

class PPMDir:
    """ 
    PPMDir is a base class for analyzing PPM run directories. This class defines
    a standard interface interacting with PPM run directories. As part of this
    interface PPMDir provides the basic attributes and methods features needed
    to analyze run directories. To analyze a spesific Run directorie format a
    class can inheret from PPMDir and define the nessisary attributes and methods
    for the spesific case.
    """
    
    _code_substring = "PPM2"
    _compile_substring = "compile"
    _jobscript_substring = "jobscript"
    _profiles_substring = ["YProfile", "RProfile"]
    _readppmin_substring = "read_ppm"
    _ftype_cycle_format = {"profile":"ftype-cycle.ext", "bobfile":"ftype-cycle.ext", "hvfile":"ftype/cycle.ext"}
    
    def __init__(self, dir="./", initalize=True, all_files=True):
        """ 
        Parameters
        ----------
        dir : string, optional
            The target directory to analyze. The default is "./".
        initalize : bool, optional
            If True the instance is initalized automaticaly, if False it must be
            initalized manualy. The default is True.
        all_files : bool, optional
            Toggles whether all files should be recorded or only recognized types
            and formats, if True all files will be recorded. The default is True
        """
        
        self._dir = dir

        self._cycles = {} # {ftype:{cycle:[paths]}}

        self._code = []
        self._compile = []
        self._jobscript = []
        self._other_files = []

        self._profiles = []
        self._bobfiles = []
        self._ppminfiles = []
        self._hvfiles = []
        
        self._all_files = all_files
        
        if initalize:
            self._initalize()
            
    def get_profile_types(self):
        """
        Get list of profile data types.
        
        Returns
        -------
        list
            Names of avaliable profile data types. 
        """
        
        return self._profiles

    def get_bobfile_types(self):
        """
        Get list of bobfile data types.
        
        Returns
        -------
        list
            Names of avaliable bobfile data types. 
        """
        
        return self._bobfiles

    def get_ppminfile_types(self):
        """
        Get list of ppminfile data types.
        
        Returns
        -------
        list
            Names of avaliable ppminfile data types. 
        """
        
        return self._ppminfiles

    def get_hvfile_types(self):
        """
        Get list of hvfile data types.
        
        Returns
        -------
        list
            Names of avaliable hvfile data types. 
        """
        
        return self._hvfiles

    def get_dumps(self, ftype):
        """
        Get list of avaliable dumps for a given data type.
        
        Parameters
        ----------
        ftype : string
            Name of data type.

        Returns
        -------
        list
            Avaliable dump numbers as a list of strings for the given data type.
        """
        
        if ftype not in self._get_dumpfile_types():
            raise ValueError("Invalid dumpfile type \"" + ftype + "\".")

        return sorted(self._cycles[ftype].keys())
        
    def get_nearest_dump(self, ftype, dump):
        if ftype not in self._get_dumpfile_types():
            raise ValueError("Invalid dumpfile type \"" + ftype + "\".")

        return min(self.get_dumps(ftype), key=lambda x:abs(int(x) - int(dump)))
    
    def get_dumpfiles(self, ftype, dump, nearest=False):
        """
        Get list of file paths to data for a given data type and dump number.
        
        Parameters
        ----------
        ftype : string
            Name of data type.
        dump : sting or integer
            Dump number.
        nearest : bool
            If True get nearest valid dump. The defalut is False.

        Returns
        -------
        list
            Paths to the data files of the type ftype for the target dump.
        """
        
        if ftype not in self._get_dumpfile_types():
            raise ValueError("Invalid dumpfile type \"" + ftype + "\".")
        
        if nearest:
            dump = self.get_nearest_dump(ftype, dump)
        
        if "{:04d}".format(int(dump)) not in self.get_dumps(ftype):
            raise ValueError("Invalid dump number \"{:04d}\".".format(int(dump)))
    
        return sorted(self._cycles[ftype]["{:04d}".format(int(dump))])

    def get_source_code(self):
        """
        Get path to source code file(s).
        
        Returns
        -------
        list
            Path to source code file(s). 
        """
        
        return self._code

    def get_compile_script(self):
        """
        Get path to compile script file(s).
        
        Returns
        -------
        list
            Path to compile script file(s). 
        """
        
        return self._compile

    def get_jobscript(self):
        """
        Get path to jobscript file(s).
        
        Returns
        -------
        list
            Path to jobscript file(s). 
        """
        
        return self._jobscript

    def get_other_files(self):
        """
        Get path to all unrecognized file(s).
        
        Returns
        -------
        list
            Path to all unrecognized file(s). 
        """
        
        return self._other_files

    def _initalize(self):
        """ 
        Analyze the directory structure and initalize the class attributes.
        """
        
        dir = os.path.abspath(self._dir) + "/"

        files = []
        for (dirpath, dirnames, filenames) in os.walk(dir):
            dirpath = (os.path.abspath(dirpath) + "/").replace(dir, "")
            for filename in filenames:
                files.append(dirpath + filename)

        files = self._load_profiles(self._dir, files)
        files = self._load_bobfiles(self._dir, files)
        files = self._load_ppminfiles(self._dir, files)
        files = self._load_hvfiles(self._dir, files)

        self._code, self._compile, self._jobscript, self._other_files = [], [], [], []
        for file in files:
            if file.startswith(self._code_substring):
                self._code.append(dir + file)
            elif file.startswith(self._compile_substring):
                self._compile.append(dir + file)
            elif file.startswith(self._jobscript_substring):
                self._jobscript.append(dir + file)
            elif self._all_files:
                self._other_files.append(dir + file)

    def _load_profiles(self, dir, files = None):
        dir = os.path.abspath(dir) + "/"

        if files is None:
            files = []
            for (dirpath, dirnames, filenames) in os.walk(dir):
                dirpath = (os.path.abspath(dirpath) + "/").replace(dir, "")
                for filename in filenames:
                    files.append(dirpath + filename)

        for ftype in self._profiles:
            if ftype in self._cycles:
                del self._cycles[ftype]
        self._profiles = []

        unused_files = []
        for file in files:
            if self._ispropath(file):
                ftype, cycle = self._ftype_cycle(file, format=self._ftype_cycle_format["profile"])
                self._profiles.append(ftype)
                if cycle.isdigit():
                    if ftype not in self._cycles:
                        self._cycles[ftype] = {}
                    if cycle not in self._cycles[ftype]:
                        self._cycles[ftype][cycle] = []
                    self._cycles[ftype][cycle].append(dir + file)
            else:
                unused_files.append(file)
        self._profiles = sorted(list(set(self._profiles)))
        return unused_files

    def _load_bobfiles(self, dir, files=None):
        dir = os.path.abspath(dir) + "/"

        if files is None:
            files = []
            for (dirpath, dirnames, filenames) in os.walk(dir):
                dirpath = (os.path.abspath(dirpath) + "/").replace(dir, "")
                for filename in filenames:
                    files.append(dirpath + filename)

        for ftype in self._bobfiles:
            if ftype in self._cycles:
                del self._cycles[ftype]
        self._bobfiles = []

        unused_files = []
        for file in files:
            if self._isbobpath(file):
                ftype, cycle = self._ftype_cycle(file, format=self._ftype_cycle_format["bobfile"])
                self._bobfiles.append(ftype)
                if cycle.isdigit():
                    if ftype not in self._cycles:
                        self._cycles[ftype] = {}
                    if cycle not in self._cycles[ftype]:
                        self._cycles[ftype][cycle] = []
                    self._cycles[ftype][cycle].append(dir + file)
            else:
                unused_files.append(file)
        self._bobfiles = sorted(list(set(self._bobfiles)))
        return unused_files

    def _load_ppminfiles(self, dir, files=None):
        dir = os.path.abspath(dir) + "/"

        if files is None:
            files = []
            for (dirpath, dirnames, filenames) in os.walk(dir):
                dirpath = (os.path.abspath(dirpath) + "/").replace(dir, "")
                for filename in filenames:
                    files.append(dirpath + filename)

        for ftype in self._ppminfiles:
            if ftype in self._cycles:
                del self._cycles[ftype]
        self._ppminfiles = []

        unused_files = []
        for file in files:
            if self._isppminpath(file):
                base, ext = os.path.splitext(os.path.basename(file))
                if ext == ".dat":
                    # if ".dat" take base with format "sak02-dump-ftype" and
                    # rearange so base is "ftype-dump"
                    base = base.split("-", 1)
                    if len(base) > 0:
                        base = base[-1]
                    base = base.split("-", 1)
                    if len(base) > 1:
                        base = base[1] + "-" + base[0]
                ftype = self._strip_numbers(base)
                self._ppminfiles.append(ftype)
                cycle = base[-4:]
                if cycle.isdigit():
                    if ftype not in self._cycles:
                        self._cycles[ftype] = {}
                    if cycle not in self._cycles[ftype]:
                        self._cycles[ftype][cycle] = []
                    self._cycles[ftype][cycle].append(dir + file)
                else:
                    cycle = "-999"
                    if ftype not in self._cycles:
                        self._cycles[ftype] = {}
                    if cycle not in self._cycles[ftype]:
                        self._cycles[ftype][cycle] = []
                    self._cycles[ftype][cycle].append(dir + file)
            else:
                unused_files.append(file)
        self._ppminfiles = sorted(list(set(self._ppminfiles)))
        return unused_files

    def _load_hvfiles(self, dir, files=None):
        dir = os.path.abspath(dir) + "/"

        if files is None:
            files = []
            for (dirpath, dirnames, filenames) in os.walk(dir):
                dirpath = (os.path.abspath(dirpath) + "/").replace(dir, "")
                for filename in filenames:
                    files.append(dirpath + filename)

        for ftype in self._hvfiles:
            if ftype in self._cycles:
                del self._cycles[ftype]
        self._hvfiles = []

        unused_files = []
        for file in files:
            if self._ishvpath(file):
                ftype, cycle = self._ftype_cycle(file, format=self._ftype_cycle_format["hvfile"])
                self._hvfiles.append(ftype)
                if cycle.isdigit():
                    if ftype not in self._cycles:
                        self._cycles[ftype] = {}
                    if cycle not in self._cycles[ftype]:
                        self._cycles[ftype][cycle] = []
                    self._cycles[ftype][cycle].append(dir + file)
            else:
                unused_files.append(file)
        self._hvfiles = sorted(list(set(self._hvfiles)))
        return unused_files

    def _ispropath(self, path):
        return False

    def _isbobpath(self, path):
        return False

    def _isppminpath(self, path):
        """ 
        Does path point to read_ppm.in or associated .dat file.
        """
        
        return (os.path.basename(path).startswith(self._readppmin_substring)) or (path.endswith("L.dat"))

    def _ftype_cycle(self, file, format="ftype-cycle.ext"):
        if format == "ftype-cycle.ext":
            base, ext = os.path.splitext(os.path.basename(file))
            ftype = self._strip_numbers(base)
            cycle = base[-4:]
            return ftype, cycle
        if format == "ftype/cycle.ext":
            base = os.path.basename(os.path.dirname(file))
            ftype = self._strip_numbers(base)
            cycle, ext = os.path.splitext(os.path.basename(file))
            return ftype, cycle

    def _ishvpath(self, path):
        return False

    def _get_dumpfile_types(self):
        """ 
        Get list of all recognized dump file types.
        """
        
        return self._profiles + self._bobfiles + self._ppminfiles + self._hvfiles

    def _strip_numbers(self, string):
        section = ""
        while len(string) > 0:
            if string[-1] in ["-", "_"]:
                string = string[:-1]
                section = ""
            elif string[-1].isdigit():
                section = string[-1] + section
                string = string[:-1]
            else:
                break
        return string + section

class old_format(PPMDir):
    """ 
    This class analyzes PPM run directories where all of the data files are
    dumped into one directory. As a submodule of PPMDir the standard PPMDir
    interface is avaliable.
    """

    def _ispropath(self, path):
        """ 
        Does path point to a profile type data file.
        """
        
        return any([path.startswith(prefix) for prefix in self._profiles_substring])

    def _isbobpath(self, path):
        """ 
        Does path point to a bobfile type data file.
        """
        
        return all([not path.startswith(profile) for profile in self._profiles_substring]) and (path[:-3].endswith(".bob") or path[:-3].endswith(".bob8"))
        
class new_format(PPMDir):
    """ 
    This class analyzes PPM run directories where the bobfiles are placed into
    seporate directories labeled by the dump number. As a submodule of PPMDir
    the standard PPMDir interface is avaliable.
    """

    def _ispropath(self, path):
        """ 
        Does path point to a profile type data file.
        """

        return any([path.startswith(prefix) for prefix in self._profiles_substring])

    def _isbobpath(self, path):
        """ 
        Does path point to a bobfile type data file.
        """

        return path[:min(4, len(path))].isdigit() and (path[:-3].endswith(".bob") or path[:-3].endswith(".bob8"))
        
class compact_format(PPMDir):
    """ 
    This class analyzes PPM run directories where the bobfiles are placed into
    seporate directories labeled by the data type and .hv files are in a the
    directory "HV/". As a submodule of PPMDir the standard PPMDir interface is
    avaliable.
    """

    _ftype_cycle_format = {"profile":"ftype/cycle.ext", "bobfile":"ftype/cycle.ext", "hvfile":"ftype/cycle.ext"}

    def _ispropath(self, path):
        """ 
        Does path point to a profile type data file.
        """

        return any([path.startswith(prefix) for prefix in self._profiles_substring])

    def _isbobpath(self, path):
        """ 
        Does path point to a bobfile type data file.
        """

        return all([not path.startswith(profile) for profile in self._profiles_substring]) and (path[:-3].endswith(".bob") or path[:-3].endswith(".bob8"))

    def _ishvpath(self, path):
        """ 
        Does path point to a hvfile type data file.
        """

        return path.startswith("HV") and path.endswith(".hv")

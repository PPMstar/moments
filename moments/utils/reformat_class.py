import shutil
import os
import subprocess
import pkg_resources
from . import fpp
import moments.core.ppmdir
import multiprocessing as mp
import sys
sys.path.insert(0, '/home/jerichoo/jupyter_py3/lib/python2.7/site-packages/')
from joblib import Parallel, delayed

__all__ = ["compact_reformat", "new_reformat", "old_reformat"]

def compact_reformat(source_dir, target_dir, all_files=False, max_dumps=None):
    """ 
    Reformat PPM run directory and convert *.bobxxx file to .hv for visualization
    
    Parameters
    ----------
    source_dir : string
        Path to the source PPM run directory.
    target_dir : string
        Path to the destination directory.
    all_files : bool, optional
        Toggle the inclusion of files not in the directory format definition.
        The default is False.
    max_dumps : integer, optional
        Limit the number of dumps to reformat. If max_dumps is None all dumps
        will be reformated. The default is None. 
    """
    #RBO: ReformatBigOutput
    if isinstance(source_dir, str):
        source_dir = os.path.abspath(source_dir) + "/"
        source = moments.core.ppmdir.get_ppmdir(source_dir, all_files)
    else:
        source = source_dir
        source_dir = source._dir

    target_dir = os.path.abspath(target_dir) + "/"
    
    profile_format = target_dir + "{ftype}/{ftype}-{dump}{ext}"
    bobfile_format = target_dir + "{ftype}/{dump}{ext}"
    ppmin_format = target_dir + "post/{fname}"
    hv_format = target_dir + "HV/{ftype}/{dump}{ext}"
    hv_processing_format = target_dir + "HV_processing/{ftype}/{RBO_in_ftype}-{dump}{ext}"
    hv_source_format = target_dir + "HV_processing/{ftype}_xreformat64_all.F"

    RBO_source = pkg_resources.resource_string("moments.utils", "/bin/ReformatBigOutputargs.F")
    RBO_compile_flags = ["-mcmodel=medium", "-i-dynamic", "-tpp7", "-xT", "-fpe0",
                         "-w", "-ip", "-Ob2", "-pc32", "-i8", "-auto", "-fpp2", "-o"]
    
    RBO_input_map = {"FVandMoms":"FVandMoms48",
                     "FV-hires":"FV-hires01",
                     "TanhUY":"TanhUY--001",
                     "TanhDivU":"TanhDivU-01",
                     "Lg10Vort":"Lg10Vort-01",
                     "Lg10ENUCbyP":"Lg10ENUCbyP"}

    RBO_default = {"isBoB8":0, "isBoB":0, "isMom":0, "isvort":0,
                   "isdivu":0, "isuy":0, "isenuc":0, "nnxteams":0,
                   "nnyteams":0, "nnzteams":0, "nntxbricks":0,
                   "nntybricks":0, "nntzbricks":0, "nnnnx":0,
                   "nnnny":0, "nnnnz":0}

    RBO_settings = {"FVandMoms":{"isMom":1},
                    "FV-hires":{"isBoB8":1},
                    "TanhUY":{"isBoB":1, "isuy":1},
                    "TanhDivU":{"isBoB":1, "isdivu":1},
                    "Lg10Vort":{"isBoB":1, "isvort":1},
                    "Lg10ENUCbyP":{"isBoB":1, "isenuc":1}}

    bob2hv_path = os.path.abspath(pkg_resources.resource_filename("moments.utils", "/bin/bob2hv"))
    
    is_hires = ["FV-hires"]

    bob2hv_input_map = {"FVandMoms":"FVandMomt48",
                        "FV-hires":"FV-hiret01",
                        "TanhUY":"TanhUY-0001",
                        "TanhDivU":"TanhDivV-01",
                        "Lg10Vort":"Lg10Voru-01",
                        "Lg10ENUCbyP":"Lg10ENVCbyP"}

    profiles = []
    bobfiles = []
    ppminfiles = []
    hvfiles = []
    
    profiles = source.get_profile_types()
    for ftype in source.get_bobfile_types():
        if "FVandMoms" in ftype:
            bobfiles.append(ftype)
        else:
            hvfiles.append(ftype)
    ppminfiles = source.get_ppminfile_types()
    
    for file in source.get_source_code() + source.get_compile_script() + source.get_jobscript() + source.get_other_files():
        fname = file.replace(source_dir, "")
        _copy_file(file, target_dir + fname)
    
    for ftype in profiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                _copy_file(file, profile_format.format(ftype=ftype, dump=dump, ext=ext))
    
    for ftype in bobfiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                _copy_file(file, bobfile_format.format(ftype=ftype, dump=dump, ext=ext))
    
    for ftype in ppminfiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                fname = os.path.basename(file)
                _copy_file(file, ppmin_format.format(fname=fname))
                
    #analyze PPM2F source code
    source_code_definitions = fpp.preprocess(source.get_source_code()[0])
    if ("nnzteams" in source_code_definitions) and ("nnxteams" not in source_code_definitions):
        source_code_definitions["nnxteams"] = source_code_definitions["nnzteams"]

    if ("nntzbricks" in source_code_definitions) and ("nntxbricks" not in source_code_definitions):
        source_code_definitions["nntxbricks"] = source_code_definitions["nntzbricks"]

    if ("nnnnz" in source_code_definitions) and ("nnnnx" not in source_code_definitions):
        source_code_definitions["nnnnx"] = source_code_definitions["nnnnz"] 
    #move files into HV_processing
    resolutionx = source_code_definitions["nnxteams"]*source_code_definitions["nntxbricks"]*source_code_definitions["nnnnx"]
    resolutiony = source_code_definitions["nnyteams"]*source_code_definitions["nntybricks"]*source_code_definitions["nnnny"]
    resolutionz = source_code_definitions["nnzteams"]*source_code_definitions["nntzbricks"]*source_code_definitions["nnnnz"]
    nteams = source_code_definitions["nnxteams"]*source_code_definitions["nnyteams"]*source_code_definitions["nnzteams"]
    print resolutionx, resolutiony, resolutionz
    
    for ftype in hvfiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                RBO_in_ftype = None
                for key, value in RBO_input_map.items():
                    if ftype in key:
                        RBO_in_ftype = value
                        break
                if RBO_in_ftype is not None:
                    nbytes = os.path.getsize(file)
                    nbytes_theoretical = (resolutionx*resolutiony*resolutionz)/nteams
                    out_file = hv_processing_format.format(ftype=ftype, RBO_in_ftype=RBO_in_ftype, dump=dump, ext=ext)
                    
                    if ftype == 'FV-hires':
                        nbytes_theoretical *= nteams # don't know if this is true
                    if nbytes != nbytes_theoretical:
                        if nbytes < nbytes_theoretical:
                            print 'this is not the right size'
                            break
                        if nbytes > nbytes_theoretical: 
                            print nbytes, '> ', nbytes_theoretical
                            out_dir = os.path.dirname(out_file)
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                                
                            truncate_command = ["tail","-c",str(nbytes_theoretical),file,">",out_file]
                            command = subprocess.Popen(' '.join(truncate_command), shell=True)
                            command.wait()
                    else:
                        _copy_file(file, out_file)
                else:
                    print 'cannot convert {ftype} files into hv'.format(ftype=ftype)

    #generate hv files
    for ftype in hvfiles:
        for RBO_key in RBO_input_map.keys():
            if ftype in RBO_key:
                settings = dict((key, source_code_definitions.get(key, value)) for key, value in RBO_default.items())
                settings.update(RBO_settings[RBO_key])

                code = fpp.define(RBO_source, **settings)
                with open(hv_source_format.format(ftype=ftype), "w") as fout:
                    fout.write(code)
                
                ftype_RBO_source = hv_source_format.format(ftype=ftype)
                ftype_RBO, _ = os.path.splitext(ftype_RBO_source)
                compile = subprocess.Popen(["ifort", ftype_RBO_source] + RBO_compile_flags + [ftype_RBO])
                compile.wait()

                for i, dump in enumerate(source.get_dumps(ftype)):
                        if max_dumps is not None:
                            if i == max_dumps:
                                break
                        
                        processing_dir = os.path.dirname(hv_processing_format.format(ftype=ftype, RBO_in_ftype="", dump="", ext=""))
                        RBO_command = [ftype_RBO, str(int(dump)), str(int(dump))]
         
                        RBO_compute = subprocess.Popen(RBO_command, cwd=processing_dir)
                        RBO_compute.wait()
        
                        bob2hv_ftype = None
                        for key, value in bob2hv_input_map.items():
                            if ftype in key:
                                bob2hv_ftype = value
                                break
                        
                        if bob2hv_ftype is not None:                
                            resolutionx = source_code_definitions["nnxteams"]*source_code_definitions["nntxbricks"]*source_code_definitions["nnnnx"]
                            resolutiony = source_code_definitions["nnyteams"]*source_code_definitions["nntybricks"]*source_code_definitions["nnnny"]
                            resolutionz = source_code_definitions["nnzteams"]*source_code_definitions["nntzbricks"]*source_code_definitions["nnnnz"]
                            print resolutionx, resolutiony, resolutionz
                            if not any([ftype in type for type in is_hires]):
                                resolutionx = int(resolutionx/2.0)
                                resolutiony = int(resolutiony/2.0)
                                resolutionz = int(resolutionz/2.0)
                                RBO_file = hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bobaaa")
                            else:
                                RBO_file = hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bob8aaa")
                               
                            bob2hv_command = [bob2hv_path, str(resolutionx), str(resolutiony), str(int(resolutionz/2.0)), RBO_file, "-t",
                                              str(source_code_definitions["nnxteams"]), str(source_code_definitions["nnyteams"]), str(2*source_code_definitions["nnzteams"]), "-s", "128"]

                            bob2hv = subprocess.Popen(bob2hv_command, cwd=processing_dir)
                            bob2hv.wait()

                            hv_file, _ = os.path.splitext(RBO_file)
                            try:
                                _move_file(hv_file + ".hv", hv_format.format(ftype=ftype, dump=dump, ext=".hv"))
                            except IOError as error:
                                print 'No .hv file was made for {ftype}'.format(ftype=hv_format.format(ftype=ftype, dump=dump, ext=".hv"))
 
    # remove all temperary files
    shutil.rmtree(os.path.dirname(hv_source_format.format(ftype="")))
 
class new_reformat(object):
    """ 
    Reformat PPM run directory and convert *.bobxxx file to .hv for visualization
    
    Parameters
    ----------
    source_dir : string
        Path to the source PPM run directory.
    target_dir : string
        Path to the destination directory.
    all_files : bool, optional
        Toggle the inclusion of files not in the directory format definition.
        The default is False.
    max_dumps : integer, optional
        Limit the number of dumps to reformat. If max_dumps is None all dumps
        will be reformated. The default is None. 
    """

    def __init__(self,source_dir,target_dir,all_files=False,max_dumps=None):
        
        #RBO: ReformatBigOutput
        self.max_dumps = max_dumps
        self.all_files = all_files

        if isinstance(source_dir, str):
            self.source_dir = os.path.abspath(source_dir) + "/"
            self.source = moments.core.ppmdir.get_ppmdir(source_dir, all_files)
        else:
            self.source = source_dir
            self.source_dir = source._dir

        self.target_dir = os.path.abspath(target_dir) + "/"

        self.profile_format = self.target_dir + "{ftype}/{ftype}-{dump}{ext}"
        self.bobfile_format = self.target_dir + "{ftype}/{dump}{ext}"
        self.ppmin_format = self.target_dir + "post/{fname}"
        self.hv_format = self.target_dir + "HV/{ftype}/{dump}{ext}"
        self.hv_processing_format = self.target_dir + "HV_processing/{ftype}/{RBO_in_ftype}-{dump}{ext}"
        self.hv_source_format = self.target_dir + "HV_processing/{ftype}_xreformat64_all.F"

        self.RBO_source = pkg_resources.resource_string("moments.utils", "/bin/ReformatBigOutputargs.F")
        self.RBO_compile_flags = ["-mcmodel=medium", "-i-dynamic", "-tpp7", "-xT", "-fpe0",
                             "-w", "-ip", "-Ob2", "-pc32", "-i8", "-auto", "-fpp2", "-o"]

        self.RBO_input_map = {"FVandMoms":"FVandMoms48",
                         "FV-hires":"FV-hires01",
                         "TanhUY":"TanhUY--001",
                         "TanhDivU":"TanhDivU-01",
                         "Lg10Vort":"Lg10Vort-01",
                         "Lg10ENUCbyP":"Lg10ENUCbyP"}

        self.RBO_default = {"isBoB8":0, "isBoB":0, "isMom":0, "isvort":0,
                       "isdivu":0, "isuy":0, "isenuc":0, "nnxteams":0,
                       "nnyteams":0, "nnzteams":0, "nntxbricks":0,
                       "nntybricks":0, "nntzbricks":0, "nnnnx":0,
                       "nnnny":0, "nnnnz":0}

        self.RBO_settings = {"FVandMoms":{"isMom":1},
                        "FV-hires":{"isBoB8":1},
                        "TanhUY":{"isBoB":1, "isuy":1},
                        "TanhDivU":{"isBoB":1, "isdivu":1},
                        "Lg10Vort":{"isBoB":1, "isvort":1},
                        "Lg10ENUCbyP":{"isBoB":1, "isenuc":1}}

        self.bob2hv_path = os.path.abspath(pkg_resources.resource_filename("moments.utils", "/bin/bob2hv"))

        self.is_hires = ["FV-hires"]

        self.bob2hv_input_map = {"FVandMoms":"FVandMomt48",
                            "FV-hires":"FV-hiret01",
                            "TanhUY":"TanhUY-0001",
                            "TanhDivU":"TanhDivV-01",
                            "Lg10Vort":"Lg10Voru-01",
                            "Lg10ENUCbyP":"Lg10ENVCbyP"}

        self.profiles = []
        self.bobfiles = []
        self.ppminfiles = []
        self.hvfiles = []


    def copy_files(self):
        self.profiles = self.source.get_profile_types()
        for ftype in self.source.get_bobfile_types():
            if "FVandMoms" in ftype:
                self.bobfiles.append(ftype)
            else:
                self.hvfiles.append(ftype)
        self.ppminfiles = self.source.get_ppminfile_types()

        for file in self.source.get_source_code() + self.source.get_compile_script() + self.source.get_jobscript() + self.source.get_other_files():
            fname = file.replace(self.source_dir, "")
            _copy_file(file, self.target_dir + fname)

        for ftype in self.profiles:
            for i, dump in enumerate(self.source.get_dumps(ftype)):
                if self.max_dumps is not None:
                    if i == self.max_dumps:
                        break
                for file in self.source.get_dumpfiles(ftype, dump):
                    base, ext = os.path.splitext(file)
                    _copy_file(file, self.profile_format.format(ftype=ftype, dump=dump, ext=ext))

        for ftype in self.bobfiles:
            for i, dump in enumerate(self.source.get_dumps(ftype)):
                if self.max_dumps is not None:
                    if i == self.max_dumps:
                        break
                for file in self.source.get_dumpfiles(ftype, dump):
                    base, ext = os.path.splitext(file)
                    _copy_file(file, self.bobfile_format.format(ftype=ftype, dump=dump, ext=ext))

        for ftype in self.ppminfiles:
            for i, dump in enumerate(self.source.get_dumps(ftype)):
                if self.max_dumps is not None:
                    if i == self.max_dumps:
                        break
                for file in self.source.get_dumpfiles(ftype, dump):
                    fname = os.path.basename(file)
                    _copy_file(file, self.ppmin_format.format(fname=fname))

        #analyze PPM2F source code
        self.source_code_definitions = fpp.preprocess(self.source.get_source_code()[0])
        if ("nnzteams" in self.source_code_definitions) and ("nnxteams" not in self.source_code_definitions):
            self.source_code_definitions["nnxteams"] = self.source_code_definitions["nnzteams"]

        if ("nntzbricks" in self.source_code_definitions) and ("nntxbricks" not in self.source_code_definitions):
            self.source_code_definitions["nntxbricks"] = self.source_code_definitions["nntzbricks"]

        if ("nnnnz" in self.source_code_definitions) and ("nnnnx" not in self.source_code_definitions):
            self.source_code_definitions["nnnnx"] = self.source_code_definitions["nnnnz"] 
        #move files into HV_processing
        self.resolutionx = self.source_code_definitions["nnxteams"]*self.source_code_definitions["nntxbricks"]*self.source_code_definitions["nnnnx"]
        self.resolutiony = self.source_code_definitions["nnyteams"]*self.source_code_definitions["nntybricks"]*self.source_code_definitions["nnnny"]
        self.resolutionz = self.source_code_definitions["nnzteams"]*self.source_code_definitions["nntzbricks"]*self.source_code_definitions["nnnnz"]
        self.nteams = self.source_code_definitions["nnxteams"]*self.source_code_definitions["nnyteams"]*self.source_code_definitions["nnzteams"]

        #pool = mp.Pool()
        #pool.map(parallel_reformat, hvfiles)
        num_cores = mp.cpu_count()

        Parallel(n_jobs=num_cores)(delayed(self.parallel_reformat)(file) for file in self.hvfiles)
        '''
        ii =0
        procs = []
        for ftype in self.hvfiles:

            for i, dump in enumerate(self.source.get_dumps(ftype)):
                if self.max_dumps is not None:
                    if i == self.max_dumps:
                        break
      
            print ftype
            p = mp.Process(target=parallel_reformat, args=(ftype))
            procs.append(p)
            p.start()
            print ii
            ii += 1
          
            #self.parallel_reformat(ftype)
        '''
        for p in procs:
            p.join()
        # remove all temperary files
        #shutil.rmtree(os.path.dirname(hv_source_format.format(ftype="")))

    def parallel_reformat(self,ftype):

        for i, dump in enumerate(self.source.get_dumps(ftype)):
            for file in self.source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                RBO_in_ftype = None
                for key, value in self.RBO_input_map.items():
                    if ftype in key:
                        RBO_in_ftype = value
                        break
                if RBO_in_ftype is not None:
                    nbytes = os.path.getsize(file)
                    nbytes_theoretical = (self.resolutionx*self.resolutiony*self.resolutionz)/self.nteams
                    out_file = self.hv_processing_format.format(ftype=ftype, RBO_in_ftype=RBO_in_ftype, dump=dump, ext=ext)

                    if ftype == 'FV-hires':
                        nbytes_theoretical *= self.nteams # don't know if this is true
                    if nbytes != nbytes_theoretical:
                        if nbytes < nbytes_theoretical:
                            print 'this is not the right size'
                            break
                        if nbytes > nbytes_theoretical: 
                            print nbytes, '> ', nbytes_theoretical
                            out_dir = os.path.dirname(out_file)
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir)

                            truncate_command = ["tail","-c",str(nbytes_theoretical),file,">",out_file]
                            command = subprocess.Popen(' '.join(truncate_command), shell=True)
                            command.wait()
                    else:
                        _copy_file(file, out_file)
                else:
                    print 'cannot convert {ftype} files into hv'.format(ftype=ftype)
                    break

            for RBO_key in self.RBO_input_map.keys():         
                if ftype in RBO_key:
                    settings = dict((key, self.source_code_definitions.get(key, value)) for key, value in self.RBO_default.items())
                    settings.update(self.RBO_settings[RBO_key])

                    code = fpp.define(self.RBO_source, **settings)
                    with open(self.hv_source_format.format(ftype=ftype), "w") as fout:
                        fout.write(code)

                    ftype_RBO_source = self.hv_source_format.format(ftype=ftype)
                    ftype_RBO, _ = os.path.splitext(ftype_RBO_source)
                    compile = subprocess.Popen(["ifort", ftype_RBO_source] + self.RBO_compile_flags + [ftype_RBO])
                    compile.wait()

                    processing_dir = os.path.dirname(self.hv_processing_format.format(ftype=ftype, RBO_in_ftype="", dump="", ext=""))
                    RBO_command = [ftype_RBO, str(int(dump)), str(int(dump))]

                    RBO_compute = subprocess.Popen(RBO_command, cwd=processing_dir)
                    RBO_compute.wait()

                    bob2hv_ftype = None
                    for key, value in self.bob2hv_input_map.items():
                        if ftype in key:
                            bob2hv_ftype = value
                            break

                    if bob2hv_ftype is not None:                
                        resolutionx = self.source_code_definitions["nnxteams"]*self.source_code_definitions["nntxbricks"]*self.source_code_definitions["nnnnx"]
                        resolutiony = self.source_code_definitions["nnyteams"]*self.source_code_definitions["nntybricks"]*self.source_code_definitions["nnnny"]
                        resolutionz = self.source_code_definitions["nnzteams"]*self.source_code_definitions["nntzbricks"]*self.source_code_definitions["nnnnz"]
                        if not any([ftype in type for type in self.is_hires]):
                            resolutionx = int(resolutionx/2.0)
                            resolutiony = int(resolutiony/2.0)
                            resolutionz = int(resolutionz/2.0)
                            RBO_file = self.hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bobaaa")
                        else:
                            RBO_file = self.hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bob8aaa")

                        bob2hv_command = [self.bob2hv_path, str(resolutionx), str(resolutiony), str(int(resolutionz/2.0)), RBO_file, "-t",
                                          str(self.source_code_definitions["nnxteams"]), str(self.source_code_definitions["nnyteams"]), str(2*self.source_code_definitions["nnzteams"]), "-s", "128"]

                        bob2hv = subprocess.Popen(bob2hv_command, cwd=processing_dir)
                        bob2hv.wait()

                        hv_file, _ = os.path.splitext(RBO_file)
                        try:
                            _move_file(hv_file + ".hv", self.hv_format.format(ftype=ftype, dump=dump, ext=".hv"))
                        except IOError as error:
                            print 'No .hv file was made for {ftype}'.format(ftype=self.hv_format.format(ftype=ftype, dump=dump, ext=".hv"))
                        
    
def old_reformat(source_dir, target_dir, all_files=False, max_dumps=None):
    if isinstance(source_dir, str):
        source_dir = os.path.abspath(source_dir) + "/"
        source = moments.core.ppmdir.get_ppmdir(source_dir, all_files)
    else:
        source = source_dir
        source_dir = source._dir

    target_dir = os.path.abspath(target_dir) + "/"
    
    profile_format = target_dir + "{ftype}-{dump}{ext}"
    bobfile_format = target_dir + "{ftype}-{dump}{ext}"
    ppmin_format = target_dir + "post/{fname}"
    
    profiles = []
    bobfiles = []
    ppminfiles = []
    
    profiles = source.get_profile_types()
    bobfiles = source.get_bobfile_types()
    ppminfiles = source.get_ppminfile_types()
    
    for file in source.get_source_code() + source.get_compile_script() + source.get_jobscript() + source.get_other_files():
        fname = file.replace(source_dir, "")
        _copy_file(file, target_dir + fname)
    
    for ftype in profiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                _copy_file(file, profile_format.format(ftype=ftype, dump=dump, ext=ext))
    
    for ftype in bobfiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                base, ext = os.path.splitext(file)
                _copy_file(file, bobfile_format.format(ftype=ftype, dump=dump, ext=ext))
    
    for ftype in ppminfiles:
        for i, dump in enumerate(source.get_dumps(ftype)):
            if max_dumps is not None:
                if i == max_dumps:
                    break
            for file in source.get_dumpfiles(ftype, dump):
                fname = os.path.basename(file)
                _copy_file(file, ppmin_format.format(fname=fname))
    
def _copy_file(source, target):
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    try:
        if not os.path.exists(target):
            shutil.copy(source, target)
    except IOError as error:
        print(error)

def _move_file(source, target):
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(target):
        shutil.move(source, target)

import shutil
import os
import subprocess
import pkg_resources
from . import fpp
import moments.core.ppmdir

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
    
    #move files into HV_processing
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
                    _copy_file(file, hv_processing_format.format(ftype=ftype, RBO_in_ftype=RBO_in_ftype, dump=dump, ext=ext))
                else:
                    ###### Put warning here: Warning cannot convert ftype files to hv ---------------
                    pass
    
    #analyze PPM2F source code
    source_code_definitions = fpp.preprocess(source.get_source_code()[0])
    if ("nnzteams" in source_code_definitions) and ("nnxteams" not in source_code_definitions):
        source_code_definitions["nnxteams"] = source_code_definitions["nnzteams"]

    if ("nntzbricks" in source_code_definitions) and ("nntxbricks" not in source_code_definitions):
        source_code_definitions["nntxbricks"] = source_code_definitions["nntzbricks"]

    if ("nnnnz" in source_code_definitions) and ("nnnnx" not in source_code_definitions):
        source_code_definitions["nnnnx"] = source_code_definitions["nnnnz"]

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
                            _move_file(hv_file + ".hv", hv_format.format(ftype=ftype, dump=dump, ext=".hv"))

    # remove all temperary files
    shutil.rmtree(os.path.dirname(hv_source_format.format(ftype="")))
    
def new_reformat(source_dir, target_dir, all_files=False, max_dumps=None):
    if isinstance(source_dir, str):
        source_dir = os.path.abspath(source_dir) + "/"
        source = moments.core.ppmdir.get_ppmdir(source_dir, all_files)
    else:
        source = source_dir
        source_dir = source._dir

    target_dir = os.path.abspath(target_dir) + "/"
    
    profile_format = target_dir + "{ftype}/{ftype}-{dump}{ext}"
    bobfile_format = target_dir + "{dump}/{ftype}/{ftype}-{dump}{ext}"
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

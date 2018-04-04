
# coding: utf-8

# In[1]:


import shutil
import os
import subprocess
import pkg_resources
import logging
import multiprocessing as mp
import sys
sys.path.insert(0, '/home/jerichoo/jupyter_py3/lib/python2.7/site-packages/')
from joblib import Parallel, delayed
sys.path.insert(0,'./moments/moments/utils')
sys.path.insert(0,'./moments/')
import moments.core.ppmdir
import fpp

log = logging.getLogger(__name__)
log.propagate = False
ch = logging.StreamHandler()
log.addHandler(ch)
log.setLevel(logging.INFO)

def reformat_hv(ftype,dump,info):
    
    #log.info('%s %s  being processed....' % (ftype, dump))
    
    if os.path.isfile(info.hv_format.format(ftype=ftype, dump=dump, ext=".hv")):
        log.info('%s %s  already has hv file overwrite=True to overwrite' % (ftype, dump))
        return
    
    for file in info.source.get_dumpfiles(ftype, dump):
        base, ext = os.path.splitext(file)
        RBO_in_ftype = None
        for key, value in info.RBO_input_map.items():
            if ftype in key:
                RBO_in_ftype = value
                break
        if RBO_in_ftype is not None:
            nbytes = os.path.getsize(file)
            nbytes_theoretical = (info.resolutionx*info.resolutiony*info.resolutionz)/info.nteams
            out_file = info.hv_processing_format.format(ftype=ftype, RBO_in_ftype=RBO_in_ftype, dump=dump, ext=ext)
            
            # code to sort out the restarts
            if ftype == 'FV-hires':
                nbytes_theoretical *= info.nteams # don't know if this is true
            if nbytes != nbytes_theoretical:
                if nbytes < nbytes_theoretical:
                    log.info('bytes: %s this is not the right size should be %s %s %s %s'                             % (nbytes, '> ', nbytes_theoretical, ftype, dump))
                    break
                if nbytes > nbytes_theoretical: 
                    log.info('bytes: %s %s %s %s %s truncated' % (nbytes, '> ', nbytes_theoretical, ftype, dump))
                    out_dir = os.path.dirname(out_file)
                    if not os.path.exists(out_dir):
                        try:
                            os.makedirs(out_dir)
                        except:
                            log.info('Trying to write to a dir while it was created') 

                    truncate_command = ["tail","-c",str(nbytes_theoretical),file,">",out_file]
                    command = subprocess.Popen(' '.join(truncate_command), shell=True)
                    command.wait()
            else:
                _copy_file(file, out_file)
        else:
            log.info('cannot convert {ftype} files into hv'.format(ftype=ftype))
            _copy_file(file, info.hv_other_format.format(ftype=ftype, RBO_in_ftype=RBO_in_ftype, dump=dump, ext=ext))
            break

    for RBO_key in info.RBO_input_map.keys():         
        if ftype in RBO_key:
            settings = dict((key, info.source_code_definitions.get(key, value)) for key, value in info.RBO_default.items())
            settings.update(info.RBO_settings[RBO_key])
            
            code = fpp.define(info.RBO_source, **settings)
            with open(info.hv_source_format.format(ftype=ftype,dump=dump), "w") as fout:
                fout.write(code)

            ftype_RBO_source = info.hv_source_format.format(ftype=ftype,dump=dump)
            ftype_RBO, _ = os.path.splitext(ftype_RBO_source)
            compile = subprocess.Popen(["ifort", ftype_RBO_source] + info.RBO_compile_flags + [ftype_RBO])
            compile.wait()

            processing_dir = os.path.dirname(info.hv_processing_format.format(ftype=ftype, RBO_in_ftype="", dump="", ext=""))
            RBO_command = [ftype_RBO, str(int(dump)), str(int(dump))]

            RBO_compute = subprocess.Popen(RBO_command, cwd=processing_dir)
            RBO_compute.wait()

            bob2hv_ftype = None
            for key, value in info.bob2hv_input_map.items():
                if ftype in key:
                    bob2hv_ftype = value
                    break

            if bob2hv_ftype is not None:                
                resolutionx = info.source_code_definitions["nnxteams"]*info.source_code_definitions["nntxbricks"]*info.source_code_definitions["nnnnx"]
                resolutiony = info.source_code_definitions["nnyteams"]*info.source_code_definitions["nntybricks"]*info.source_code_definitions["nnnny"]
                resolutionz = info.source_code_definitions["nnzteams"]*info.source_code_definitions["nntzbricks"]*info.source_code_definitions["nnnnz"]
                if not any([ftype in type for type in info.is_hires]):
                    resolutionx = int(resolutionx/2.0)
                    resolutiony = int(resolutiony/2.0)
                    resolutionz = int(resolutionz/2.0)
                    RBO_file = info.hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bobaaa")
                else:
                    RBO_file = info.hv_processing_format.format(ftype=ftype, RBO_in_ftype=bob2hv_ftype, dump=dump, ext=".bob8aaa")
            
                bob2hv_command = [info.bob2hv_path, str(resolutionx), str(resolutiony), str(int(resolutionz/2.0)), RBO_file, "-t",
                                  str(info.source_code_definitions["nnxteams"]), str(info.source_code_definitions["nnyteams"]), str(2*info.source_code_definitions["nnzteams"]), "-s", "128"]

                bob2hv = subprocess.Popen(bob2hv_command, cwd=processing_dir)
                bob2hv.wait()

                hv_file, _ = os.path.splitext(RBO_file)
                try:
                    _move_file(hv_file + ".hv", info.hv_format.format(ftype=ftype, dump=dump, ext=".hv"))
                except IOError as error:
                    log.error('No .hv file was made for {ftype}'.format(ftype=info.hv_format.format(ftype=ftype, dump=dump, ext=".hv")))
                    log.error('Either {RBO} or {bob} command failed'.format(RBO=RBO_command,bob =bob2hv_command))
                    
                    
def _copy_file(source, target):
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except:
            log.info('Trying to write to a dir while it was created') 
    try:
        if not os.path.exists(target):
            shutil.copy(source, target)
            pass
    except IOError as error:
        print(error)

def _move_file(source, target):
    target_dir = os.path.dirname(target)
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except:
            log.info('Trying to write to a dir while it was created') 
    if not os.path.exists(target):
        shutil.move(source, target)

class reformation_info(object):
    
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
        '''
        self.profile_format = self.target_dir + "{ftype}/{ftype}-{dump}{ext}"
        self.bobfile_format = self.target_dir + "{ftype}/{dump}{ext}" #chan
        self.ppmin_format = self.target_dir + "post/{fname}"
        self.hv_format = self.target_dir + "HV/{ftype}/{dump}{ext}"
        self.hv_processing_format = self.target_dir + "HV_processing/{ftype}/{RBO_in_ftype}-{dump}{ext}"
        self.hv_source_format = self.target_dir + "HV_processing/{ftype}{dump}_xreformat64_all.F"
        '''
        self.profile_format = self.target_dir + "{ftype}/{ftype}-{dump}{ext}"
        self.bobfile_format = self.target_dir + "{ftype}/{dump}/{ftype}-{dump}{ext}" #chan
        self.ppmin_format = self.target_dir + "post/{fname}"
        self.hv_format = self.target_dir + "HV/{ftype}/{ftype}-{dump}{ext}"
        self.hv_processing_format = self.target_dir + "HV_processing/{ftype}/{RBO_in_ftype}-{dump}{ext}"
        self.hv_other_format = self.target_dir + "other_bob/{ftype}/{RBO_in_ftype}-{dump}{ext}"
        self.hv_source_format = self.target_dir + "HV_processing/{ftype}{dump}_xreformat64_all.F"

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

        #analyze PPM2F source code
        for file in self.source.get_source_code():
            if file.endswith(source_dir[-2:]+'.F'):
                log.info(file, ' will be used as the source file')
                source_code = file
                break
                
        if 'source_code' not in locals():
            source_code = self.source.get_source_code()[0]
            log.info('%s will be used as the source file' % (file))    
        self.source_code_definitions = fpp.preprocess(source_code)
        if ("nnzteams" in self.source_code_definitions) and ("nnxteams" not in self.source_code_definitions):
            self.source_code_definitions["nnxteams"] = self.source_code_definitions["nnzteams"]

        if ("nntzbricks" in self.source_code_definitions) and ("nntxbricks" not in self.source_code_definitions):
            self.source_code_definitions["nntxbricks"] = self.source_code_definitions["nntzbricks"]

        if ("nnnnz" in self.source_code_definitions) and ("nnnnx" not in self.source_code_definitions):
            self.source_code_definitions["nnnnx"] = self.source_code_definitions["nnnnz"] 
        #move files into HV_processing
        self.resolutionx = self.source_code_definitions["nnxteams"]*self.source_code_definitions["nntxbricks"]            *self.source_code_definitions["nnnnx"]
        self.resolutiony = self.source_code_definitions["nnyteams"]*self.source_code_definitions["nntybricks"]            *self.source_code_definitions["nnnny"]
        self.resolutionz = self.source_code_definitions["nnzteams"]*self.source_code_definitions["nntzbricks"]            *self.source_code_definitions["nnnnz"]
        self.nteams = self.source_code_definitions["nnxteams"]*self.source_code_definitions["nnyteams"]            *self.source_code_definitions["nnzteams"]
        
        print self.resolutionx, self.nteams
        
        self.profiles = self.source.get_profile_types()
        for ftype in self.source.get_bobfile_types():
            if "FVandMoms" in ftype:
                self.bobfiles.append(ftype)
            else:
                self.hvfiles.append(ftype)
        self.ppminfiles = self.source.get_ppminfile_types()
    
def copy_source_code(info,file):
    
    #for file in info.source.get_source_code() + info.source.get_compile_script() + info.source.get_jobscript() + info.source.get_other_files():
    fname = file.replace(info.source_dir, "")
    _copy_file(file, info.target_dir + fname)
        
def copy_profiles(info,ftype,dump):
    
    for file in info.source.get_dumpfiles(ftype, dump):
        base, ext = os.path.splitext(file)
        _copy_file(file, info.profile_format.format(ftype=ftype, dump=dump, ext=ext))
                
def copy_bobfiles(info,ftype,dump):
    
    for file in info.source.get_dumpfiles(ftype, dump):
        base, ext = os.path.splitext(file)
        _copy_file(file, info.bobfile_format.format(ftype=ftype, dump=dump, ext=ext))
                
def copy_ppminfiles(info):
    
    for ftype in info.ppminfiles:
        for i, dump in enumerate(info.source.get_dumps(ftype)):
            if info.max_dumps is not None:
                if i == info.max_dumps:
                    break
            for file in info.source.get_dumpfiles(ftype, dump):
                fname = os.path.basename(file)
                _copy_file(file, info.ppmin_format.format(fname=fname))

def reformat_parallel(info,target_dir,all_files=False,max_dumps=None):
    
    #setting up a logger
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logs_dir = target_dir + '/reformatting_log'
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    fh = logging.FileHandler(os.path.join(logs_dir, 'log_' + target_dir[-2:]))
    fh.setFormatter(formatter)
    log.addHandler(fh)
    
    # copy_only(info)
    other_files = (
                    info.source.get_source_code() 
                    + info.source.get_compile_script() 
                    + info.source.get_jobscript()
                    + info.source.get_other_files())
    
    copy_ppminfiles(info)
    
    num_cores = mp.cpu_count() #this always returns 32
    
    # Running the copy process in parallel
    Parallel(n_jobs=num_cores)(
        delayed(copy_source_code)(info,file) for file in other_files)
   
    Parallel(n_jobs=num_cores)(
        delayed(copy_profiles)(info,ftype,ndump) for ftype in info.profiles\
        for ndump in info.source.get_dumps(ftype))
    
    Parallel(n_jobs=num_cores)(
        delayed(copy_bobfiles)(info,ftype,ndump) for ftype in info.bobfiles\
        for ndump in info.source.get_dumps(ftype))
    
    # Running the hv transformation in parallel
    Parallel(n_jobs=num_cores)(
        delayed(reformat_hv)(ftype,ndump,info) for ftype in info.hvfiles\
        for ndump in info.source.get_dumps(ftype))
                        
    os.system('rm -rf {}'.format(os.path.dirname(info.hv_source_format.format(ftype="",dump=""))))


for name in [3,4,5]:
    
    try:
        in_dir = '/home/jerichoo/projects/rrg-fherwig-ad/fherwig/PPM_unprocessed/B/PPM_B{}'.format(name)
        out_dir = '/scratch/jerichoo/B/pPPM_B{}'.format(name)

        info = reformation_info(in_dir,out_dir,all_files=True,max_dumps=None)

        reformat_parallel(info,out_dir,all_files = True)
    except:
        print 'no reformatting done for B {}'.format(name)


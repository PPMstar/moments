################################################################################
#                                                                              #
# ppmfield.py - A python module                                                #
#                                                                              #
# (c) 2016 Luke Siemens                                                        #
#                                                                              #
################################################################################

"""
ppmfield.py

ppmfield is a Python module for performing mathematical operations on PPM data.
This module uses numpy and wrapper to create a hybrid array like data type to
perform the mathamatical operations.
"""

__all__ = ["PPMField", "array", "dot", "norm", "radprof"]

import numpy
import sys

def array(object):
    """ 
    Create an array which is compatible with e3d vector and tensor fields.
    
    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, or any (nested) sequence.

    Returns
    -------
    output : numpy.ndarray or PPMField
        An array object satisfying the specified requirements.
    """

    try:
        if _isppmfield(object):
            return object.view(PPMField)
        elif all(_isppmfield(element) for element in object):
            if len(object) > 1:
                if all(_arecompatible(object[i], object[i + 1]) for i in range(len(object) - 1)):
                    for i in range(len(object) - 1):
                        for j in range(i + 1, len(object)):
                            if object[i]._field == object[j]._field:
                                #if duplicate fields create alternate
                                object[j] = object[j]._alternate_field()
                    wrapper = object[0]._wrapper
                    statement = " ".join([element._field for element in object])
                    shape = (len(object), wrapper._xresolution, wrapper._yresolution, wrapper._zresolution)
                    return PPMField(wrapper=wrapper, statement=statement, children=object, shape=shape)
                else:
                    return numpy.array([element.toarray() for element in object])
            else:
                return numpy.array([object[0].toarray()])
    except IndexError:
        pass
                
    return numpy.array(object)

def dot(a, b):
    """ 
    Dot product of two arrays, compatible with e3d vector and tensor fields.
    
    This operation is defined as ::
    
        dot(a, b)[i,j,k] = sum(a[:,i,j,k] * b[:,i,j,k])
        
    Parameters
    ----------
    a : numpy.ndarray or PPMField
        First field.
    b : numpy.ndarray or PPMField
        Second field.
    
    Returns
    -------
    output : numpy.ndarray or PPMField
        An array object.
    """

    if _isppmfield(a) and _isppmfield(b):
        if _arecompatible(a, b):
            wrapper = a._wrapper
            statement = "dot(" + a._field + ", " + b._field + ")"
            shape = (wrapper._xresolution, wrapper._yresolution, wrapper._zresolution)
            return PPMField(wrapper=wrapper, statement=statement, children=[a, b], shape=shape)        
        
    return numpy.sum(a*b, axis=0)

def norm(field):
    """ 
    Get the Euclidean norm of the vector field.
    
    Parameters
    ----------
    Field : numpy.ndarray or PPMField
        A vector field.
    
    Returns
    -------
    magnitude : numpy.ndarray or PPMField
        A scalar field.
    """
    
    if _isppmfield(field):
        wrapper = field._wrapper
        statement = "norm(" + field._field + ")"
        shape = (wrapper._xresolution, wrapper._yresolution, wrapper._zresolution)
        return PPMField(wrapper=wrapper, statement=statement, children=[field], shape=shape)
        
    return numpy.sqrt(numpy.sum(field**2, axis=0))
    
def radprof(field, statistics=False):
    """ 
    Computes the 3-D radial profile of the field.
    
    Parameters
    ----------
    field : numpy.ndarray or PPMField
        Input field.
    statistics : bool, optional
        Toggles radprof statistics. The default is False.
    
    Returns
    -------
    output : numpy.ndarray or tuple
        The radial profile of the given field. If statistics is False only the
        radial profile is returned. If statistics is True then in addition to
        the radial profile the standard deviation, minimum and maximum profiles
        are also returned, the order is (radprof, standard deviation, min, max).
    """

    if _isppmfield(field):
        if not statistics:
            return field._get_radprof()[:, 0]
        else:
            return numpy.swapaxes(field._get_radprof(), 0, 1)
    
    if len(field.shape) > 3:
        raise NotImplementedError("Tensor fields not supported.")

    xfull = field.shape[0]

#    profile = numpy.zeros(shape=(int(xfull/2),), dtype=numpy.float32)
    X, Y, Z = numpy.indices(field.shape) - 0.5*(xfull - 1)
    ird = numpy.sqrt(X*X + Y*Y + Z*Z).astype(dtype=numpy.int32)

    var = numpy.bincount(ird.ravel(), field.ravel(), minlength=int(xfull/2))[:int(xfull/2)]
    num = numpy.bincount(ird.ravel(), minlength=int(xfull/2))[:int(xfull/2)]

    if not statistics:
        return var/num
    else:
        var2 = numpy.bincount(ird.ravel(), numpy.square(field).ravel(), minlength=int(xfull/2))[:int(xfull/2)]
    
        disp = numpy.sqrt(var2/num - numpy.square(var/num))
        prof_min = numpy.zeros(shape=(int(xfull/2),), dtype=numpy.float32)
        prof_max = numpy.zeros(shape=(int(xfull/2),), dtype=numpy.float32)
        
        for i in range(int(xfull/2)):
            prof_min[i] = field[ird == i].min()
            prof_max[i] = field[ird == i].max()
        return var/num, disp, prof_min, prof_max

class PPMField(numpy.ndarray):
    """ 
    PPMField is data type for computing derived fields from PPM FVandMoms48 data.
    It attemps to interpret mathematical statements, functions and algorithms
    written in python or using numpy and evaluate them using the e3d command line
    tool. If an operation which has not been implemented acts on a PPMField
    instance then it will automaticaly evaluate itself upto that point and pass
    the result as a numpy.ndarray to the original operation.

    Parameters
    ----------
    wrapper : moments.core.Wrapper, optional
        A Wrapper instance, the default is None. The default is None.
    statement : string
        Definition of the field, either the name of a field in FVandMoms48
        data or a statement defining the field in terms of other fields.
        The default is None.
    children : list, optional
        A list of PPMField instances which define the fields used in statement.
        The default is [].
    shape : tuple
        Shape of the field given as a tuple of integers. The default is (0, 0, 0).

    Returns
    -------
    field : PPMField
        A PPMField instance satisfying the specified requirements.

    Notes
    -----
    If wrapper or statement is None the instance will be treated as invalid.
    """        

    _current_id = 0
    _fields = ["FV", "DivUav", "Vortmag", "Prs", "Rho", "RhoUx", "RhoUy", "RhoUz",
                        "RhoUxUx", "RhoUxUy", "RhoUxUz", "RhoUyUy", "RhoUyUz", "RhoUzUz",
                        "RhoHux", "RhoHuy", "RhoHuz"]
    
    def __new__(cls, wrapper=None, statement=None, children=[], shape=(0, 0, 0)):
        """ 
        Parameters
        ----------
        wrapper : moments.core.Wrapper, optional
            A Wrapper instance, the default is None. The default is None.
        statement : string
            Definition of the field, either the name of a field in FVandMoms48
            data or a statement defining the field in terms of other fields.
            The default is None.
        children : list, optional
            A list of PPMField instances which define the fields used in statement.
            The default is [].
        shape : tuple
            Shape of the field given as a tuple of integers. The default is (0, 0, 0).

        Returns
        -------
        field : PPMField
            A PPMField instance satisfying the specified requirements.

        Notes
        -----
        If wrapper or statement is None the instance will be treated as invalid.
        """
        
        # ppmfiled.__new__ is called when explicitly initalizing a new instance of PPMField
#        print("In __new__ of class: " + str(cls) + "\n\tcalling numpy.ndarray.__new__\n")

        # initalize underlying ndarray componet as a one element array set to numpy.nan
        self = numpy.ndarray.__new__(cls, shape=shape, dtype=numpy.float32, order="F")
        self[:] = numpy.nan
        
        # initalize PPMField component of self
        PPMField._current_id += 1
        if statement in self._fields:
            self._field = str(statement)
            self._statement = ""
        else:
            self._field = "field" + str(PPMField._current_id)
            self._statement = self._field + " = " + str(statement)
        self._children = children
        self._wrapper = wrapper
        self._id = PPMField._current_id
        return self
    
    def __array_finalize__(self, obj):
        # PPMField.__array_finalize__ is allways called when an instance of PPMField is created
        # if obj is None then PPMField.__new__ has allready been called
        # else initalize self from obj which may or may not be an instance of PPMField
#        print("In __array_finalize__\n\tself: " + str(type(self)) + "\n\tobj: " + str(type(obj)) + "\n")

        if obj is None:
            pass
        else:
            #initalize self from obj
            PPMField._current_id += 1
            self._field = getattr(obj, "_field", None)
            self._statement = getattr(obj, "_statement", None)
            self._children = getattr(obj, "_children", None)
            self._wrapper = getattr(obj, "_wrapper", None)
            self._id = PPMField._current_id
            
    def __array_wrap__(self, array, context=None):
        """ 
        Intercept calls to numpy universal functions (ufunc).
        """
        
        # PPMField.__array_wrap__ is called on the return of a ufunc
#        print("In __array_wrap__\n\tself: " + repr(self) + "\n\tarray: " + repr(array) + "\n\tcontext:" + str(context) + "\n")
        
        if context is not None:
            # ufunc instance, tupple of inputs to ufunc, index of array in output tupple
            ufunc, arguments, output_index = context
            if len(arguments) == 1:
                # evalue unary ufuncs
                arg, = arguments

                try: # using try except since the PPMField math operator functions can also raise NotImplementedError
                    if not _isppmfield(arg):
                        # not valid PPMField
                        raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                    
                    # -------- Apply unary operator -------- #
                    operation = ""
                    if ufunc == numpy.negative:
                        numpy.multiply(-1.0, arg, out=array)
                        return array
                    elif ufunc == numpy.absolute:
                        operation = "abs(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.exp:
                        operation = "exp(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.log:
                        operation = "log(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.log10:
                        operation = "log10(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.sqrt:
                        operation = "sqrt(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.sin:
                        operation = "sin(" + array._e3d_repr(arg) + ")"
                    elif ufunc == numpy.cos:
                        operation = "cos(" + array._e3d_repr(arg) + ")"
                    else:
                        raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                        
                    return array._operator(operation, arg)
                except NotImplementedError:
                    return ufunc(arg.toarray())
                
            elif len(arguments) == 2:
                # evaluate binary ufuncs
                arg1, arg2 = arguments
                
                try: # using try except since the PPMField math operator functions can also raise NotImplementedError
                    if not _arecompatible(arg1, arg2):
                        # one of arg1, arg2 are in valid or arg1 and arg2 are incompatible
                        raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                    
                    if not _isppmfield(arg1):
                        if hasattr(arg1, "__len__"):
                            arg1 = numpy.array(arg1)
                            if len(arg1.tolist()) != 1:
                                # cannot boadcast arg1
                                raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                            else:
                                arg1 = arg1.tolist()[0]
                        arg1 = _constant_field(self, arg1)
    
                    if not _isppmfield(arg2):
                        if hasattr(arg2, "__len__"):
                            arg2 = numpy.array(arg2)
                            if len(arg2.tolist()) != 1:
                                # cannot boadcast arg2
                                raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                            else:
                                arg2 = arg2.tolist()[0]
                        arg2 = _constant_field(self, arg2)
    
                    # -------- Apply binary operator -------- #
                    operation = ""
                    if ufunc == numpy.add:
                        operation = array._e3d_repr(arg1) + " + " + array._e3d_repr(arg2)
                    elif ufunc == numpy.subtract:
                        operation = array._e3d_repr(arg1) + " - " + array._e3d_repr(arg2)
                    elif ufunc == numpy.multiply:
                        operation = array._e3d_repr(arg1) + "*" + array._e3d_repr(arg2)
                    elif (ufunc == numpy.divide) or (ufunc == numpy.true_divide):
                        operation = array._e3d_repr(arg1) + "/" + array._e3d_repr(arg2)
                    else:
                        raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")
                        
                    return array._operator(operation, arg1, arg2)
                except NotImplementedError:
                    if _isppmfield(arg1):
                        arg1 = arg1.toarray()

                    if _isppmfield(arg2):
                        arg2 = arg2.toarray()
                    return ufunc(arg1, arg2)
        else:
            raise NotImplementedError("numpy ufunc with context: " + str(context) + " not implimented")

    def __repr__(self):
        base = numpy.ndarray.__repr__(self.toarray())
        return base[:-1] + ", field=" + str(self._field) + ", formula=\"" + str(self._statement) + "\", children=[" + ", ".join([str(child._field) + ".id=" + str(child._id) for child in self._children]) + "], _id=" + str(self._id) + ", ownsData=" + str(self.base is None) + ")"

    def __getslice__(self, i, j):
        if i == 0:
            i = None
        if j == sys.maxsize:
            j = None
        return self.__getitem__(slice(i, j, None))

    def __getitem__(self, item):
        try:
            if all([element == slice(None, None, None) for element in item]):
                return numpy.ndarray.__getitem__(self, item)
        except TypeError:
            if item == slice(None, None, None):
                return numpy.ndarray.__getitem__(self, item)
        
        return numpy.ndarray.__getitem__(self.toarray(), item)

    def toarray(self):
        """ 
        Cast the field as a numpy.ndarray.
        
        Returns
        -------
        output : numpy.ndarray
            The result of evaluating the field.
        """
        
        return self._get_bof()
    
    def _e3d_repr(self, arg):
        """ 
        Get the field identifier or string repersentation of arg.
        """
        
        if _isppmfield(arg):
            return str(arg._field)
        else:
            return str(arg)
                
    def _compile_statement(self, defined=[]):
        """ 
        Get full set of equations defining the field.
        """
        
        statement = ""
        for child in self._children:
            if not child._field in defined:
                child_statement, defined = child._compile_statement(defined)
                if len(child_statement) > 0:
                    statement = statement + child_statement + "\n"
        
        return (statement + self._statement, defined + [self._field])
    
    def _get_bof(self):
        """ 
        Generate and load bof file of field.
        """
        
        if len(self.shape) > 3:
            raise NotImplementedError("Generating tensor fields not implemented.")
        
        formulas, _ = self._compile_statement()
        self._wrapper.process_bobfiles()
        self._wrapper.generate_bof(self._field, formulas=formulas)
        return self._wrapper.read_bof(self._wrapper._outfile[:-3] + self._field + ".bof")
    
    def _get_radprof(self):
        """ 
        Generate and load radial profile of field.
        """
        
        if len(self.shape) > 3:
            raise NotImplementedError("Generating tensor fields not implemented.")
        
        formulas, _ = self._compile_statement()
        self._wrapper.process_bobfiles()
        self._wrapper.generate_radprof(self._field, formulas=formulas)
        return self._wrapper.read_radprof(self._wrapper._outfile[:-3] + self._field + ".radprof")

    def _alternate_field(self):
        """
        Duplicate the field under a new field identifier.
        """
        
        if self._statement != "":
            field = PPMField.view(self)
            field._initialize_operator()
            field._statement = field._field + " =" + self._statement.split("=", 1)[-1]
            field._children = self._children
            field._wrapper = self._wrapper
        else:
            field = 1.0*self
        return field

    def _coord(self, axis):
        """ 
        Get coordinate field for given axis.
        
        Parameters
        ----------
        axis : string
            Name of coordinate.
        """
        
        field = PPMField.view(self)
        field._initialize_operator()
        if axis == "x":
            field._statement = field._field + " = coord1(" + field._e3d_repr(self) + ")"
        elif axis == "y":
            field._statement = field._field + " = coord2(" + field._e3d_repr(self) + ")"
        elif axis == "z":
            field._statement = field._field + " = coord3(" + field._e3d_repr(self) + ")"
        field._children = [self]
        field._wrapper = self._wrapper
        return field

    def _initialize_operator(self):
        """ 
        Initalize parameters.
        """
        
        PPMField._current_id += 1
        self._field = "field" + str(PPMField._current_id)
        self._statement = ""
        self._children = []
        self._id = PPMField._current_id
    
    def _finalize_operator(self, *args):
        """ 
        Set field children and wrapper instance
        
        Parameters
        ----------
        args : PPMField
            PPMField instances.
        """
        
        for arg in args:
            if _isppmfield(arg):
                self._children.append(arg)
                self._wrapper = arg._wrapper

    def _operator(self, operation, *args):
        """ 
        Get new PPMField defined as a function of other fields
        
        Parameters
        ----------
        operation : string
            Definition of field interms of other instances.
        args : PPMField
            Fields used in definition.
        """
        
        self._initialize_operator()
        self._statement = self._field + " = " + operation
        self._finalize_operator(*args)
        return self

def _constant_field(field_instance, const):
    """
    Get constant field
    
    Parameters
    ----------
    field_instance : PPMField
        PPMField instance to supply wrapper object.
    const : float or integer
        Value of the constant field.
    """
    
    if _isppmfield(field_instance):
        wrapper = field_instance._wrapper
        statement = "set_constant(" + str(const) + ")"
        shape = (wrapper._xresolution, wrapper._yresolution, wrapper._zresolution)
        return PPMField(wrapper=wrapper, statement=statement, children=[], shape=shape)
    else:
        raise valueError("field_instance must be a valid PPMField.")

def _isppmfield(arg):
    """ 
    Check if arg is a valid PPMField.
    """
    
    try:
        field_arg = getattr(arg, "_field")
        wrapper_arg = getattr(arg, "_wrapper")
        compile_statement_arg = getattr(arg, "_compile_statement")

        if (field_arg is None) or (wrapper_arg is None) or (compile_statement_arg is None):
            return False
    except AttributeError:
        return False        
    return True

def _arecompatible(arg1, arg2):
    """ 
    Check if arg1 and arg2 are compatible PPMField instances.
    """
    
    try:
        field_arg1 = getattr(arg1, "_field")
        wrapper_arg1 = getattr(arg1, "_wrapper")
        compile_statement_arg1 = getattr(arg1, "_compile_statement")

        if (field_arg1 is None) or (wrapper_arg1 is None) or (compile_statement_arg1 is None):
            return False
    except AttributeError:
        wrapper_arg1 = None
    
    try:
        field_arg2 = getattr(arg2, "_field")
        wrapper_arg2 = getattr(arg2, "_wrapper")
        compile_statement_arg2 = getattr(arg2, "_compile_statement")

        if (field_arg2 is None) or (wrapper_arg2 is None) or (compile_statement_arg2 is None):
            return False
    except AttributeError:
        wrapper_arg2 = None

    if (wrapper_arg1 is not None) and (wrapper_arg2 is not None):
        if wrapper_arg1 != wrapper_arg2:
            return False
    
    return True

import unittest
import numpy
import moments
from matplotlib import pyplot

data_dir = "/npod2/users/lsiemens/PPM/D15"
show_error_plots = False

numpy_rtol = 2.5e-7
numpy_atol = 1.0e-9

cycle = 20

def one_args_field(function):
    def Test(self):
        field = function(self, self.Rho)
        ndarr = function(self, self.Rho_ndarr)
        try:
            numpy.testing.assert_allclose(field, ndarr, rtol=numpy_rtol, atol=numpy_atol)
        except AssertionError:
            if not show_error_plots:
                raise
            else:
                pyplot.imshow((field - ndarr)[int(len(ndarr)/2.0),:,:])
                pyplot.show()
                radprof, rad_disp, rad_min, rad_max = moments.radprof(field - ndarr, statistics=True)
                tol, tol_disp, tol_min, tol_max = moments.radprof(numpy_atol + numpy_rtol*numpy.absolute(field), statistics=True)
                pyplot.errorbar(self.data.raxis, radprof, yerr=rad_disp, label="Field profile, 1 sigma err")
                pyplot.plot(self.data.raxis, rad_max, label="Field maximum")
                pyplot.errorbar(self.data.raxis, tol, yerr=tol_disp, label="Tolerance profile, 1 sigma err")
                pyplot.plot(self.data.raxis, tol_max, label="Tolerance maximum")
                pyplot.legend(loc=0)
                pyplot.show()
                raise
    return Test

def two_args_field(function):
    def Test(self):
        field = function(self, self.RhoUx, self.Rho)
        ndarr = function(self, self.RhoUx_ndarr, self.Rho_ndarr)
        try:
            numpy.testing.assert_allclose(field, ndarr, rtol=numpy_rtol, atol=numpy_atol)
        except AssertionError:
            if not show_error_plots:
                raise
            else:
                pyplot.imshow((field - ndarr)[int(len(ndarr)/2.0),:,:])
                pyplot.show()
                radprof, rad_disp, rad_min, rad_max = moments.radprof(field - ndarr, statistics=True)
                tol, tol_disp, tol_min, tol_max = moments.radprof(numpy_atol + numpy_rtol*numpy.absolute(field), statistics=True)
                pyplot.errorbar(self.data.raxis, radprof, yerr=rad_disp, label="Field profile, 1 sigma err")
                pyplot.plot(self.data.raxis, rad_max, label="Field maximum")
                pyplot.errorbar(self.data.raxis, tol, yerr=tol_disp, label="Tolerance profile, 1 sigma err")
                pyplot.plot(self.data.raxis, tol_max, label="Tolerance maximum")
                pyplot.legend(loc=0)
                pyplot.show()
                raise
    return Test

def three_args_field(function):
    def Test(self):
        field = function(self, self.RhoUx, self.RhoUy, self.RhoUz)
        ndarr = function(self, self.RhoUx_ndarr, self.RhoUy_ndarr, self.RhoUz_ndarr)
        try:
            numpy.testing.assert_allclose(field, ndarr, rtol=numpy_rtol, atol=numpy_atol)
        except AssertionError:
            if not show_error_plots:
                raise
            else:
                pyplot.imshow((field - ndarr)[int(len(ndarr)/2.0),:,:])
                pyplot.show()
                radprof, rad_disp, rad_min, rad_max = moments.radprof(field - ndarr, statistics=True)
                tol, tol_disp, tol_min, tol_max = moments.radprof(numpy_atol + numpy_rtol*numpy.absolute(field), statistics=True)
                pyplot.errorbar(self.data.raxis, radprof, yerr=rad_disp, label="Field profile, 1 sigma err")
                pyplot.plot(self.data.raxis, rad_max, label="Field maximum")
                pyplot.errorbar(self.data.raxis, tol, yerr=tol_disp, label="Tolerance profile, 1 sigma err")
                pyplot.plot(self.data.raxis, tol_max, label="Tolerance maximum")
                pyplot.legend(loc=0)
                pyplot.show()
                raise
    return Test

class TestMath(unittest.TestCase):
    def setUp(self):
        self.data = moments.Moments(data_dir)
        self.FV = self.data.get("FV", cycle, nearest=True)
        self.Rho = self.data.get("Rho", cycle, nearest=True)
        self.RhoUx = self.data.get("RhoUx", cycle, nearest=True)
        self.RhoUy = self.data.get("RhoUy", cycle, nearest=True)
        self.RhoUz = self.data.get("RhoUz", cycle, nearest=True)

        self.FV_ndarr = self.FV.toarray()
        self.Rho_ndarr = self.Rho.toarray()
        self.RhoUx_ndarr = self.RhoUx.toarray()
        self.RhoUy_ndarr = self.RhoUy.toarray()
        self.RhoUz_ndarr = self.RhoUz.toarray()
    
    @three_args_field
    def test_dot_repeated_fields(self, A, B, C):
        array = moments.array([A, B, C])
        return moments.dot(array, array)

    @three_args_field
    def test_dot(self, A, B, C):
        array_one = moments.array([A, B, C])
        array_two = moments.array([C, B, A])
        return moments.dot(array_one, array_two)

    @three_args_field
    def test_norm(self, A, B, C):
        array = moments.array([A, B, C])
        return moments.norm(array)
        
    @one_args_field
    def test_negative_python(self, A):
        return -A

    @one_args_field
    def test_negative_numpy(self, A):
        return numpy.negative(A)
    
    @one_args_field
    def test_abs_python(self, A):
        return abs(A)

    @one_args_field
    def test_abs_numpy(self, A):
        return numpy.absolute(A)
        
    @one_args_field
    def test_exp_numpy(self, A):
        #using -abs(A) as argument to avoid overflow error due to massive exponets
        return numpy.exp(-abs(A))
    
    @one_args_field
    def test_log_numpy(self, A):
        return numpy.log(A)    

    @one_args_field
    def test_log10_numpy(self, A):
        return numpy.log10(A)

    @one_args_field
    def test_sqrt_numpy(self, A):
        return numpy.sqrt(A)

    @one_args_field
    def test_sin_numpy(self, A):
        return numpy.sin(A)

    @one_args_field
    def test_cos_numpy(self, A):
        return numpy.cos(A)

    @two_args_field
    def test_add_python(self, A, B):
        return A + B

    @two_args_field
    def test_add_numpy(self, A, B):
        return numpy.add(A, B)

    @two_args_field
    def test_subtract_python(self, A, B):
        return A - B

    @two_args_field
    def test_subtract_numpy(self, A, B):
        return numpy.subtract(A, B)

    @two_args_field
    def test_multiply_python(self, A, B):
        return A*B

    @two_args_field
    def test_multiply_numpy(self, A, B):
        return numpy.multiply(A, B)

    @two_args_field
    def test_divide_python(self, A, B):
        return A/B

    @two_args_field
    def test_divide_numpy(self, A, B):
        return numpy.divide(A, B)

if __name__ == "__main__":
    unittest.main()

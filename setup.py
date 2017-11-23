from setuptools import setup, find_packages

setup(
    name = "moments",
    version = "0.1",
    packages = find_packages(),
    package_data = {"moments.utils": ["bin/*"]},
    install_requires = ["numpy", "setuptools"],
    tests_require=["nose"],
    test_suite="nose.collector",
    author = "Luke Siemens",
    author_email = "lsiemens@uvic.ca")

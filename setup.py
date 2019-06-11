#!/usr/bin/env python

"""
Standard python setup.py file for MLaaS4HEP
"""
import os
import sys
import fnmatch
import subprocess

from distutils.core import setup

def datafiles(dir, pattern=None):
    """Return list of data files in provided relative dir"""
    files = []
    for dirname, dirnames, filenames in os.walk(dir):
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))
        for filename in filenames:
            if  filename[-1] == '~':
                continue
            # match file name pattern (e.g. *.css) if one given
            if pattern and not fnmatch.fnmatch(filename, pattern):
                continue
            files.append(os.path.join(dirname, filename))
    return files

def main():
    "Main setup"
    # this try/except block is taking care of setting up proper
    # version either by reading it from MLaaS4HEP package
    # this part is used by pip to get the package version
    try:
        import MLaaS4HEP
        version = MLaaS4HEP.__version__
    except:
        version = '0.0.0'
        init = 'src/python/MLaaS4HEP/__init__.py'
        if os.path.exists(init):
            with open(init) as istream:
                for line in istream.readlines():
                    line = line.replace('\n', '')
                    if line.startswith('__version__'):
                        version = str(line.split('=')[-1]).strip().replace("'", "").replace('"', '')
    ver = sys.version.split(' ')[0]
    pver = '.'.join(ver.split('.')[:-1])
    lpath = 'lib/python{}/site-packages'.format(pver)
    dist = setup(
        name                 = 'MLaaS4HEP',
        version              = version,
        author               = 'Valentin Kuznetsov',
        author_email         = 'vkuznet@gmail.com',
        license              = 'MIT',
        description          = 'MLaaS framework for HEP',
        long_description     = 'MLaaS framework for HEP',
        packages             = ['MLaaS4HEP'],
        package_dir          = {'MLaaS4HEP': 'src/python/MLaaS4HEP'},
        install_requires     = ['keras', 'numpy', 'pandas', 'uproot', 'pyarrow'],
        scripts              = ['bin/%s'%s for s in os.listdir('bin')],
        url                  = 'https://github.com/dmwm/MLaaS4HEP',
        data_files           = [],
        classifiers          = [
            "Programming Language :: Python",
            "Operating System :: OS Independent",
            "License :: OSI Approved :: MIT License",
            ],
    )

def read_init(init_file):
    "Read package init file and return its content"
    init = None
    with open(init_file) as istream:
        init = istream.read()
    return init

def write_version(init_file, init_content):
    "Write package init file with given content"
    if not init_content:
        init_content = \
"""
__version__ = '0.0.0'
__all__ = []
"""
    if init_file:
        with open(init_file, 'w') as ostream:
            ostream.write(init_content)

def version():
    "Return git tag version of the package or custom version"
    cmd = 'git tag --list | tail -1'
    ver = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
    ver = str(ver.decode("utf-8")).replace('\n', '')
    ver = ver if ver else '0.0.0'
    return ver

if __name__ == "__main__":
    # This part is used by `python setup.py sdist` to build package tar-ball
    # read git version
    ver = version()
    # read package init file
    init_file = 'src/python/MLaaS4HEP/__init__.py'
    init = read_init(init_file)
    # replace package init file with our version
    write_version(init_file, init.replace('0.0.0', ver))
    # execute setup main
    main()
    # put back original content of package init file
    write_version(init_file, init.replace(ver, '0.0.0'))

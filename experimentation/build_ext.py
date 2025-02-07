# build_ext.py

import os
import sys
import sysconfig
import subprocess
import importlib.util

def compile_extension():
    """
    Compiles payment_schedule.cpp into a shared library that Python can import.
    Returns the name of the compiled module file.
    """
    # Determine the extension suffix (e.g., .cp312-win_amd64.pyd)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    module_filename = f"payment_schedule{ext_suffix}"

    # Collect the pybind11 include flags
    py_executable = sys.executable
    try:
        pybind11_includes = subprocess.check_output(
            [py_executable, "-m", "pybind11", "--includes"]
        ).decode().strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error retrieving pybind11 include flags") from e

    # Path to the Python 'libs' directory (where python312.lib is located)
    python_libs_path = r"C:\Users\KumarAkashdeep\AppData\Local\Programs\Python\Python312\libs"
    
    # On Windows with MinGW, linking explicitly against python312.lib is required
    # -L tells g++ where to look for .lib files
    # -lpython312 links against python312.lib
    compile_cmd = (
        f'g++ -O3 -Wall -shared -std=c++14 '          # Compiler flags
        f'{pybind11_includes} '                      # pybind11 includes
        f'payment_schedule.cpp '                     # Your source file
        f'-L"{python_libs_path}" -lpython312 '       # Link to python312.lib
        f'-o {module_filename}'
    )

    print("Compiling the C++ extension module with command:")
    print(compile_cmd)

    subprocess.run(compile_cmd, shell=True, check=True)
    print("Compilation finished successfully.")

    return module_filename

def import_extension(module_filename):
    """
    Dynamically imports the compiled C++ module given its filename.
    Returns the imported module.
    """
    module_path = os.path.abspath(module_filename)
    spec = importlib.util.spec_from_file_location("payment_schedule", module_path)
    if spec is None:
        raise ImportError("Could not load spec for the module.")
    payment_schedule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(payment_schedule)
    return payment_schedule

def build_and_import():
    """
    Compiles the extension and imports it.
    Returns the imported module.
    """
    module_filename = compile_extension()
    return import_extension(module_filename)

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
    # Determine the proper extension suffix.
    # On Windows, python3-config is not available so we use sysconfig.
    if os.name == 'nt':
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    else:
        ext_suffix = subprocess.check_output(["python3-config", "--extension-suffix"]).decode().strip()

    module_filename = f"payment_schedule{ext_suffix}"

    # Get the include flags for pybind11 (using the current python executable).
    py_executable = sys.executable  # This ensures we use the same Python interpreter.
    try:
        pybind11_includes = subprocess.check_output(
            [py_executable, "-m", "pybind11", "--includes"]
        ).decode().strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Error retrieving pybind11 include flags") from e

    # Choose the compiler and flags based on the operating system.
    if os.name == 'nt':
        # On Windows, assume g++ is available (e.g., from MinGW or a similar toolchain).
        # Note: Adjust the command as needed for your Windows environment.
        compiler = "g++"
        # Windows usually does not need -fPIC.
        compile_cmd = (
            f'{compiler} -O3 -Wall -shared -std=c++14 '
            f'{pybind11_includes} payment_schedule.cpp -o {module_filename}'
        )
    else:
        # On Unix-like systems.
        compiler = "c++"
        compile_cmd = (
            f'{compiler} -O3 -Wall -shared -std=c++14 -fPIC '
            f'{pybind11_includes} payment_schedule.cpp -o {module_filename}'
        )

    print("Compiling the C++ extension module with command:")
    print(compile_cmd)

    # Run the compile command. shell=True is used here for command-line expansion.
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

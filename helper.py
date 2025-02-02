
import cProfile
import pstats
import io
from functools import wraps
import logging
import psutil
import time
import os

from config.settings.base import DISKSTORE_PATH

def completion_percent(completed_so_far, val_date_filtered,chunk_index,num_splits,product_variant_name):
    percent_complete = (completed_so_far / len(val_date_filtered)) * 100
    logging.warning(f"Processing chunk {chunk_index}/{num_splits} for {product_variant_name}")
    logging.warning(f"Number of positions processed: {completed_so_far}, percentage complete: {percent_complete:.6f}%")

CSV_FILE = f'{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Profile/computation_engine_profiling.txt'
csv_columns = ['Function', 'Execution Time (s)', 'CPU Usage (%)']

# Take a snapshot of resource utilization at the beginning and at the end of the process
def end2end_monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
          
        cpu_before = psutil.cpu_percent(interval=None)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        cpu_after = psutil.cpu_percent(interval=None) 

        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()

        logging.info(f"Function: {func.__name__}")
        logging.info(s.getvalue())
        logging.info(f"Execution time: {end_time - start_time} seconds")
        logging.info(f"CPU usage: {cpu_after - cpu_before}%")
        # Add 2 more lines to print memory utilization in the code base

        # Append information directly to the CSV file
        with open(CSV_FILE, mode='a', newline='') as file:
            file.write(f"{func.__name__},{end_time - start_time},{cpu_after - cpu_before}\n")
            # Add 2 more lines to append memory utilization in the written file

        return result
    return wrapper

# Simple Line profiling of the fxn 
from line_profiler import LineProfiler
def profile_function(func):
    def wrapper(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        function_name = func.__name__

        output_filename = f'{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Profile/line_profiler_output_{function_name}.txt'
        with open(output_filename, 'w') as f:
            profiler.print_stats(stream=f)

        return result
    return wrapper


#EXP USE-CASE
# Duration Monitor keep track of resource utilisation over a set time period
def continue_monitor(duration_minutes=5, csv_filename="computation_engine_profiling.csv"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            
            user_accepted_pathname = "{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Profile"
            csv_path = os.path.join(user_accepted_pathname, csv_filename)

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            csv_columns = ['Function', 'Execution Time (s)', 'CPU Usage (%)', 'Memory Usage' ]

            if not os.path.exists(csv_path):
                with open(CSV_FILE, mode='a', newline='') as file:
                    file.write(f"{func.__name__},{time_elapsed},{cpu_usage}\n")

            pr = cProfile.Profile()
            pr.enable()

            cpu_before = psutil.cpu_percent(interval=None)
            mem_before = psutil.virtual_memory().used

            start_time = time.time()
            result = func(*args, **kwargs)

            while True:
                time_elapsed = time.time() - start_time
                cpu_usage = psutil.cpu_percent(interval=None)
                mem_usage = psutil.virtual_memory().used

                with open(CSV_FILE, mode='a', newline='') as file:
                    file.write(f"{func.__name__},{time_elapsed},{cpu_usage}\n")
                # Add 2 more lines to append memory utilization in the written file

                if time_elapsed >= duration_minutes * 60:
                    break

            pr.disable()

            cpu_after = psutil.cpu_percent(interval=None)
            mem_after = psutil.virtual_memory().used
            end_time = time.time()
            with open(CSV_FILE, mode='a', newline='') as file:
                file.write(f"{func.__name__},{start_time - end_time},{cpu_after}\n")

            # log additional information 

            return result
        return wrapper
    return decorator

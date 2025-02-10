from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import time
import pyarrow.dataset as ds
import tracemalloc


def convert_to_arrow(df):
    start_time = time.time()
    convert_to_arrow = {'id': "int64[pyarrow]", 'entity': "string[pyarrow]", 'date': "date64[pyarrow]", 'amount': 'float64[pyarrow]', 'rate': 'float64[pyarrow]'}
    df_arrow = df.astype(convert_to_arrow)
    end_time = time.time()
    conversion_time = end_time - start_time
    return df_arrow, conversion_time

def load_excel(sheet_name):
    start_time = time.time()
    df = pd.read_excel('C:\\Users\\ShinTiemLee\\Downloads\\test_data.xlsx', sheet_name=sheet_name)
    end_time = time.time()
    return df, end_time - start_time

def measure_pyarrow_operations(df_arrow, df2_arrow):
    times = {}
    memory_usage = {}

    # Squaring rate column
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    squared_rate = pc.multiply(pa.array(df_arrow['rate']), pa.array(df_arrow['rate']))
    df_arrow['rate'] = squared_rate.to_pandas()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['square'] = end_time - start_time
    memory_usage['square'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Comparing DataFrames
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    df_arrow_diff = df_arrow.compare(df2_arrow)
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['compare'] = end_time - start_time
    memory_usage['compare'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Arithmetic operations
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = pc.subtract(pa.array(df_arrow['amount']), pc.multiply(pa.array(df_arrow['rate']), pa.scalar(10)))
    temp = pc.add(temp, pc.divide(pa.array(df_arrow['amount']), pa.scalar(100)))
    temp = temp.to_pandas()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['arithmetic'] = end_time - start_time
    memory_usage['arithmetic'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Filtering with conditions
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    amount_greater = pc.greater(pa.array(df_arrow['amount']), pa.scalar(15000, type=pa.float64()))
    rate_greater_equal = pc.greater_equal(pa.array(df_arrow['rate']), pa.scalar(20, type=pa.float64()))
    combined_filter = pc.and_(amount_greater, rate_greater_equal)
    combined_filter = combined_filter.to_pandas()
    temp = df_arrow[combined_filter]
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['filter'] = end_time - start_time
    memory_usage['filter'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Conditional operation
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    condition = pc.greater(pa.array(df_arrow['rate']), pa.scalar(1, type=pa.float64()))
    temp = pc.if_else(condition, pc.multiply(pa.array(df_arrow['rate']), pa.array(df_arrow['amount'])), pa.array(df_arrow['rate']))
    temp = temp.to_pandas()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['conditional'] = end_time - start_time
    memory_usage['conditional'] = snapshot2.compare_to(snapshot1, 'lineno')

    # String slicing
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = pa.compute.utf8_slice_codeunits(pa.array(df_arrow['entity']), 0, 3)
    temp = temp.to_pandas()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['string_slice'] = end_time - start_time
    memory_usage['string_slice'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Updating rate based on conditions
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    entity_condition = pc.equal(pa.array(df_arrow['entity']), pa.scalar('AAAAAA_1', type=pa.string()))
    date_condition = pc.less(pa.array(df_arrow['date']), pa.scalar(datetime.strptime('2025-02-20', '%Y-%m-%d'), type=pa.date64()))
    combined_condition = pc.and_(entity_condition, date_condition)
    combined_filter = combined_condition.to_pandas()
    df_arrow.loc[combined_filter, 'rate'] = 0.01
    df_arrow['rate'] = df_arrow['rate'].astype(float)  # Ensure correct dtype
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['update'] = end_time - start_time
    memory_usage['update'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Group by and sum
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df_arrow.groupby('entity')['amount'].sum()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['groupby_sum'] = end_time - start_time
    memory_usage['groupby_sum'] = snapshot2.compare_to(snapshot1, 'lineno')

    return times, memory_usage

def measure_pandas_operations(df, df2):
    times = {}
    memory_usage = {}

    # Squaring rate column
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    df['rate'] = df['rate'] ** 2
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['square'] = end_time - start_time
    memory_usage['square'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Comparing DataFrames
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    df_diff = df.compare(df2)
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['compare'] = end_time - start_time
    memory_usage['compare'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Arithmetic operations
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df['amount'] - df['rate'] * 10 + df['amount'] / 100
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['arithmetic'] = end_time - start_time
    memory_usage['arithmetic'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Filtering with conditions
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df.loc[(df['amount'] > 15000) & (df['rate'] >= 20)]
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['filter'] = end_time - start_time
    memory_usage['filter'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Conditional operation
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df.apply(lambda x: x['rate'] * x['amount'] if x['rate'] > 1 else x['amount'], axis=1)
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['conditional'] = end_time - start_time
    memory_usage['conditional'] = snapshot2.compare_to(snapshot1, 'lineno')

    # String slicing
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df['entity'].str[0:3]
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['string_slice'] = end_time - start_time
    memory_usage['string_slice'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Updating rate based on conditions
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    df.loc[(df['entity'].astype(str) == 'AAAAAA_1') & (df['date'] < datetime.strptime('2025-02-20', '%Y-%m-%d')), 'rate'] = 0.01
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['update'] = end_time - start_time
    memory_usage['update'] = snapshot2.compare_to(snapshot1, 'lineno')

    # Group by and sum
    start_time = time.time()
    snapshot1 = tracemalloc.take_snapshot()
    temp = df.groupby('entity')['amount'].sum()
    end_time = time.time()
    snapshot2 = tracemalloc.take_snapshot()
    times['groupby_sum'] = end_time - start_time
    memory_usage['groupby_sum'] = snapshot2.compare_to(snapshot1, 'lineno')

    return times, memory_usage

def main():
    tracemalloc.start()
    
    pyarrow_times = []
    pandas_times = []
    pyarrow_memory = []
    pandas_memory = []

    df, time = load_excel('Sheet1')
    print('Time to load Sheet1:', time)

    df2, time = load_excel('Sheet2')
    print('Time to load Sheet2:', time)

    df_arrow, conversion_time = convert_to_arrow(df)
    print(f"Time to convert the datatypes of Sheet1: {conversion_time} seconds")
    df2_arrow, conversion_time = convert_to_arrow(df2)
    print(f"Time to convert the datatypes of Sheet2: {conversion_time} seconds")
    
    for _ in range(3):
        times, memory_usage = measure_pyarrow_operations(df_arrow, df2_arrow)
        pyarrow_times.append(times)
        pyarrow_memory.append(memory_usage)
        times, memory_usage = measure_pandas_operations(df, df2)
        pandas_times.append(times)
        pandas_memory.append(memory_usage)

    # Calculate average times
    avg_pyarrow_times = {key: sum(d[key] for d in pyarrow_times) / len(pyarrow_times) for key in pyarrow_times[0]}
    avg_pandas_times = {key: sum(d[key] for d in pandas_times) / len(pandas_times) for key in pandas_times[0]}

    # Create DataFrame with results
    results = pd.DataFrame([avg_pyarrow_times, avg_pandas_times], index=['PyArrow', 'Pandas']).T
    results['Operation'] = results.index
    results['PyArrow Faster'] = results['PyArrow'] < results['Pandas']
    results['PyArrow Faster'] = results['PyArrow Faster'].apply(lambda x: 'Yes' if x else 'No')

    print("Time Benchmark Results:")
    print(results)

    # Format memory usage results
    memory_results = []
    for operation in pyarrow_memory[0].keys():
        pyarrow_memory_avg = sum(stat.size for mem in pyarrow_memory for stat in mem[operation]) / len(pyarrow_memory)
        pandas_memory_avg = sum(stat.size for mem in pandas_memory for stat in mem[operation]) / len(pandas_memory)
        memory_results.append({
            'Operation': operation,
            'PyArrow': pyarrow_memory_avg,
            'Pandas': pandas_memory_avg,
            'PyArrow Less Memory': pyarrow_memory_avg < pandas_memory_avg
        })

    memory_df = pd.DataFrame(memory_results)
    memory_df['PyArrow Less Memory'] = memory_df['PyArrow Less Memory'].apply(lambda x: 'Yes' if x else 'No')

    print("Memory Usage Benchmark Results:")
    print(memory_df)

main()
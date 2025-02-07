import logging

import time
import json
import multiprocessing
import numpy as np
import pandas as pd
import subprocess
import platform
import re
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


# from Core.users.computations.db_centralised_function import read_data_func, data_handling
from config import DISKSTORE_PATH


from Valuation_Models import Valuation_Models
import pyarrow as pa
import pyarrow.parquet as pq
import random
from numba import float64, guvectorize, int64, njit, void
from datetime import date, datetime
from multiprocessing import Pool
from helper import completion_percent


def holiday_code_generator(product_data_row, weekday_data):
    if product_data_row["weekend_definition"] != "None":
        holiday_weekends = list(json.loads(product_data_row["weekend_definition"]).keys())
    else:
        holiday_weekends = "None"
    weekday_data = json.loads(weekday_data)
    weekday_dataframe = pd.DataFrame(weekday_data)
    if holiday_weekends != "None":
        holidays = []
        for i in holiday_weekends:
            holiday = weekday_dataframe.loc[weekday_dataframe["id"] == int(i), "day"].iloc[0]
            holidays.append(holiday)
        business_days = ""
        for j in weekday_data["day"]:
            if j in holidays:
                business_days += "0"
            else:
                business_days += "1"
    else:
        holidays = []
        business_days = "1111100"

    product_variant_name = product_data_row["product_variant_name"]
    z_spread_calculation = product_data_row["z_spread_calculation"]
    product_data_df = pd.DataFrame(
        [
            {
                "product_variant_name": product_variant_name,
                "weekend": holidays,
                "business_days": business_days,
                "z_spread_calculation": z_spread_calculation,
            }
        ]
    )
    return product_data_df

def curve_component_transformation(curve_repo_data_ind):
    curve_components = {"curve_components": list(json.loads(curve_repo_data_ind["curve_components"]).keys())}
    components_new_df = pd.DataFrame(curve_components)
    components_new_df["curve_components"] = components_new_df["curve_components"].astype("int")
    components_new_df["curve_name"] = curve_repo_data_ind["curve_name"]
    components_new_df["interpolation_algorithm"] = curve_repo_data_ind["interpolation_algorithm"]
    components_new_df["extrapolation_algorithm"] = curve_repo_data_ind["extrapolation_algorithm"]
    components_new_df["day_count"] = curve_repo_data_ind["day_count"]
    if "compounding_frequency_output" in curve_repo_data_ind.keys():
        components_new_df["compounding_frequency_output"] = curve_repo_data_ind[
            "compounding_frequency_output"
        ]
    del curve_components
    return components_new_df


def daycount_convention_code(self, daycount_convention):
    if isinstance(daycount_convention, (list, np.ndarray)):
        daycount_convention = daycount_convention[0]
    
    daycount_convention_lower = daycount_convention.lower()
    daycount_mapping = {
        "30/360_bond_basis": 1,
        "30/360": 1,
        "30/360_us": 2,
        "30e/360": 3,
        "30e/360_isda": 4,
        "30e+/360_isda": 5,
        "act/360": 6,
        "act/365": 7,
        "actual/365": 7,
        "act/365l": 8,
        "act/365a": 9,
        "nl/365": 10,
        "act/act_isda": 11,
        "act/act_icma": 12,
        "business/252": 13,
        "act/act": 14,
        "actual/actual": 14,
    }
    
    # Return the code based on the mapping, default to the input if not found
    return daycount_mapping.get(daycount_convention_lower, daycount_convention)


def discount_rate_calc(ttm,tenor,rates,interpolation_algorithm,extrapolation_algorithm):
    if interpolation_algorithm == "Linear":
        calculated_rate = linearinterp(tenor, rates, np.float64(ttm))
        if calculated_rate is None:
            if extrapolation_algorithm == "Linear":
                calculated_rate = linearexterp(tenor, rates, np.float64(ttm))
            else:
                calculated_rate = flatexterp(tenor, rates, np.float64(ttm))
        else:
            calculated_rate = calculated_rate
    return calculated_rate

def discount_factor_calculation_general(row, interest_rate_col):
    if row["curve_compounding_frequency"] in ["Continuous", "continuous", "continuously"]:
        df = np.exp(-row["time_to_maturity"] * row[interest_rate_col])
    elif row["curve_compounding_frequency"] in ["monthly", "Monthly"]:
        df = np.power(1 + row[interest_rate_col] / 12, -12 * row["time_to_maturity"])
    elif row["curve_compounding_frequency"] in ["quarterly", "Quarterly"]:
        df = np.power(1 + row[interest_rate_col] / 4, -4 * row["time_to_maturity"])
    elif row["curve_compounding_frequency"] in ["semi-annualised", "Semi-Annual", "semi-annually"]:
        df = np.power(1 + row[interest_rate_col] / 2, -2 * row["time_to_maturity"])
    elif row["curve_compounding_frequency"] in ["bi-annual", "Bi-Annual", "bi-annually"]:
        df = np.power(1 + row[interest_rate_col] / 0.5, -0.5 * row["time_to_maturity"])
    else:
        df = np.power(1 + row[interest_rate_col], -row["time_to_maturity"])
    return df

# linear interpolation
@njit(cache=True, fastmath=True)
def linearinterp(x, y, independent_var_value):
    n = len(x)
    for j in range(1, n):
        if (x[j - 1]) <= independent_var_value <= (x[j]):
            return y[j - 1] + ((y[j] - y[j - 1]) * (independent_var_value - x[j - 1]) / (x[j] - x[j - 1]))


# linear extrapolation
@njit(cache=True, fastmath=True)
def linearexterp(x, y, independent_var_value):
    if independent_var_value > (x[-1]):
        return y[-1] + (independent_var_value - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    elif independent_var_value < x[0]:
        return y[0] + (independent_var_value - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


# cubic spline
def cubicspline(x, y, independent_var_value):
    check = CubicSpline(x, y)
    return check(independent_var_value)


@njit(cache=True, fastmath=True)
def flatexterp(x, y, independent_var_value):
    if independent_var_value > (x[-1]):
        return y[-1]
    elif independent_var_value < x[0]:
        return y[0]


@guvectorize([void(float64[:], float64[:], float64, float64[:])], "(n),(n),()->()")
def linforward(x, y, independent_var_value, result):
    n = len(x)
    for j in range(1, n):
        if (x[j - 1]) < independent_var_value < (x[j]):
            forward0 = (x[j] * y[j] - x[j - 1] * y[j - 1]) / (x[j] - x[j - 1])
            forward1 = (x[j + 1] * y[j + 1] - x[j] * y[j]) / (x[j + 1] - x[j])
            interpolated_forward = forward0 + (
                (forward1 - forward0) * (independent_var_value - x[j - 1]) / (x[j] - x[j - 1])
            )

            result[:] = (
                y[j - 1] * x[j - 1] + interpolated_forward * (independent_var_value - x[j - 1])
            ) / independent_var_value


# piecewise linear
@njit(cache=True, fastmath=True)
def bilinearint(x, y, f, x0, y0):
    w = []
    n = len(x)
    xu = min(c for c in x if c >= x0)
    xl = max(d for d in x if d <= x0)
    yu = min(g for g in y if g >= y0)
    yl = max(h for h in y if h <= y0)
    for j in range(0, n):
        if xl <= x[j] <= xu and yl <= y[j] <= yu:
            w.append([x[j], y[j], f[j]])
    return (
        (w[0][2] * (w[2][0] - x0) * (w[1][1] - y0))
        + (w[2][2] * (x0 - w[0][0]) * (w[1][1] - y0))
        + (w[1][2] * (w[2][0] - x0) * (y0 - w[0][1]))
        + (w[3][2] * (x0 - w[0][0]) * (y0 - w[0][1]))
    ) / ((w[2][0] - w[0][0]) * (w[1][1] - w[0][1]))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return super().default(obj)

def cashflow_dict_generation(position_id, cashflow_output_data):
    cashflow_dict = json.dumps(cashflow_output_data.loc[cashflow_output_data['position_id']==position_id].replace("", "-").fillna("-").to_dict("records"), cls=NpEncoder)
    return cashflow_dict

def sensitivity_dict_generation(sensitivity_output_data):
    sensitivity_dict = json.dumps([{"Macaulay Duration":sensitivity_output_data["Macaulay Duration"],"Modified Duration":sensitivity_output_data["Modified Duration"],"PV01 per unit":sensitivity_output_data["PV01 per unit"]}])
    return sensitivity_dict

# When CPU usage exceeds 80%, trashing (excessive paging) may occur, leading to performance degradation.
cpu_total_count = int(multiprocessing.cpu_count() * 0.8)
func = Valuation_Models.Value_extraction_pf

def preprocess_position_data(position_data, column_index_dict, reporting_date, cashflow_uploaded_data): 
    
    date_columns = [col for col in position_data.columns if re.search(r'_date$', col)]
    for col in date_columns:
        if col in column_index_dict:
            position_data[col] = pd.to_datetime(position_data[col], errors='coerce')
    
    if 'base_rate' in column_index_dict:
        position_data['base_rate'] = position_data['base_rate'].replace(['', 'nan', 'None'], np.nan).fillna(0).astype(float) / 100
    if 'fixed_spread' in column_index_dict:
        position_data['fixed_spread'] = position_data['fixed_spread'].replace(['', 'nan', 'None'], np.nan).fillna(0).astype(float) / 100
    if 'outstanding_amount' in column_index_dict:
        position_data['outstanding_amount'] = position_data['outstanding_amount'].replace(['', 'nan', 'None'], np.nan).fillna(0).astype(float)
    if 'quantity' in column_index_dict:
        position_data['quantity'] = position_data['quantity'].replace(['', 'nan', 'None'], np.nan).fillna(1).astype(float)
    
    position_data["primary_currency"] = position_data["primary_currency"].astype(str)
    position_data["reporting_currency"] = position_data["reporting_currency"].astype(str)
    position_data["secondary_currency"] = position_data["secondary_currency"].astype(str)

    # Identify positions to skip based on cashflow_uploaded_data and npa_flag
    position_data['model_code'] = np.where(position_data['npa_flag'] == 1, "M085", position_data['model_code'])
    positions_with_uploaded_cashflow = position_data['position_id'].isin(cashflow_uploaded_data['position_id'])
    positions_with_npa_flag = position_data['npa_flag'] == 1
    position_data['fixed_or_float_flag'] = np.where(position_data['fixed_or_float_flag'] == "M079", "Fixed", position_data['fixed_or_float_flag'])
    
    # Combine conditions to identify positions to skip
    positions_to_skip = positions_with_uploaded_cashflow | positions_with_npa_flag
    skipped_positions = position_data[positions_to_skip].drop_duplicates()
    position_data_to_process = position_data[~positions_to_skip]
    
    # General Rules with Exceptions
    general_error_rules = [
        {'condition': (position_data_to_process['outstanding_amount'] <= 0) & (~position_data_to_process['model_code'].isin(['M075','M005','M001'])), 'message': 'outstanding_amount <= 0'},
        {'condition': position_data_to_process['maturity_date'] < reporting_date, 'message': 'maturity_date < reporting_date'},
        {'condition': position_data_to_process['last_payment_date'] > reporting_date, 'message': 'last_payment_date > reporting_date'},
        {'condition': position_data_to_process['next_payment_date'] < reporting_date, 'message': 'next_payment_date < reporting_date'},
    ]
    
    # Model-Specific Rules
    model_specific_error_rules = [
        {'condition': (position_data_to_process['model_code'] == "M075") & (position_data_to_process['unutilized_limit_amount'] <= 0), 'message': 'unutilized_limit_amount <= 0'},
        {'condition': (position_data_to_process['model_code'] == "M020") &  (position_data_to_process['emi_amount'] <= 0), 'message': 'emi_amount <= 0'},
    ]
    
    overall_rules = general_error_rules + model_specific_error_rules

    error_position_data = pd.DataFrame()
    for rule in overall_rules:
        error_rows = position_data_to_process[rule['condition']].copy()
        error_rows['reason_for_drop'] = rule['message']
        error_position_data = pd.concat([error_position_data, error_rows])
        count_dropped = len(error_rows)
        #### " dropped due to {rule['message']} count : {count_dropped}"

    # Exclude error rows from processable data
    processable_position_data = position_data_to_process[~position_data_to_process.index.isin(error_position_data.index)]
    error_position_data = error_position_data[['product_variant_name', 'model_code', 'unique_reference_id', 'reason_for_drop']]
    
    # Add back the skipped positions (avoiding duplicates)
    processable_position_data = pd.concat([processable_position_data, skipped_positions]).drop_duplicates()

    
    return processable_position_data, error_position_data


def worker_init(
    config_dict,
    column_index_dict,
    vol_repo_data,
    vol_components_data,
    holiday_calendar,
    currency_data,
    NMD_adjustments,
    repayment_schedule,
    market_data,
    vix_data,
    cf_analysis_id,
    cashflow_uploaded_data,
    underlying_position_data,
    custom_daycount_conventions,
    dpd_ruleset,
    overdue_bucketing_data,
    dpd_schedule,
    product_holiday_code,
    request,
    curve_data,
    credit_spread_data,
):
    # Initialize global variables in each worker process to reduce data transfer overhead
    global G_config_dict, G_column_index_dict, G_vol_repo_data, G_vol_components_data
    global G_holiday_calendar, G_currency_data, G_NMD_adjustments, G_repayment_schedule
    global G_market_data, G_vix_data, G_cf_analysis_id, G_cashflow_uploaded_data
    global G_underlying_position_data, G_custom_daycount_conventions, G_dpd_ruleset
    global G_overdue_bucketing_data, G_dpd_schedule, G_product_holiday_code, G_request
    global G_curve_data, G_credit_spread_data

    G_config_dict = config_dict
    G_column_index_dict = column_index_dict
    G_vol_repo_data = vol_repo_data
    G_vol_components_data = vol_components_data
    G_holiday_calendar = holiday_calendar
    G_currency_data = currency_data
    G_NMD_adjustments = NMD_adjustments
    G_repayment_schedule = repayment_schedule
    G_market_data = market_data
    G_vix_data = vix_data
    G_cf_analysis_id = cf_analysis_id
    G_cashflow_uploaded_data = cashflow_uploaded_data
    G_underlying_position_data = underlying_position_data
    G_custom_daycount_conventions = custom_daycount_conventions
    G_dpd_ruleset = dpd_ruleset
    G_overdue_bucketing_data = overdue_bucketing_data
    G_dpd_schedule = dpd_schedule
    G_product_holiday_code = product_holiday_code
    G_request = request
    G_curve_data = curve_data
    G_credit_spread_data = credit_spread_data



def worker_func(row):
    """
    Worker function to process a single row.
    Returns:
      ("success", result) on success or
      ("failure", unique_reference_id, error_message) on exception.
    """
    try:
        # Call the passed function 'func' with the necessary global parameters
        result = func(
            row,
            G_column_index_dict,
            G_config_dict,
            G_vol_repo_data,
            G_vol_components_data,
            G_holiday_calendar,
            G_currency_data,
            G_NMD_adjustments,
            G_repayment_schedule,
            G_market_data,
            G_vix_data,
            G_cf_analysis_id,
            G_cashflow_uploaded_data,
            G_underlying_position_data,
            G_custom_daycount_conventions,
            G_dpd_ruleset,
            G_overdue_bucketing_data,
            G_dpd_schedule,
            G_product_holiday_code,
            G_request,
            G_curve_data,
            G_credit_spread_data,
        )
        unique_reference_id = row[G_column_index_dict["unique_reference_id"]]
        logging.warning(f"SUCCESS for Unique_Reference_Id: {unique_reference_id}")
        return ("success", result)
    except Exception as e:
        unique_reference_id = row[G_column_index_dict["unique_reference_id"]]
        logging.warning(
            f"ERROR for Unique_Reference_Id {unique_reference_id} occurred: {e}"
        )
        # Return only the unique_reference_id and the error message
        return ("failure", unique_reference_id, str(e))


def applyParallel(
    config_dict,
    column_index_dict,
    pos_data,
    vol_repo_data,
    vol_components_data,
    holiday_calendar,
    currency_data,
    NMD_adjustments,
    repayment_schedule,
    func,
    vix_data,
    cf_analysis_id,
    cashflow_uploaded_data,
    underlying_position_data,
    custom_daycount_conventions,
    dpd_ruleset,
    overdue_bucketing_data,
    dpd_schedule,
    product_holiday_code,
    request,
    market_data,
    curve_data,
    credit_spread_data,
):
    """
    Applies the worker_func to all positions in parallel.
    On failure, only stores the unique_reference_id and the error message.
    """
    logging.warning("Multiprocessing start")

    with Pool(
        processes=cpu_total_count,
        initializer=worker_init,
        initargs=(
            config_dict,
            column_index_dict,
            vol_repo_data,
            vol_components_data,
            holiday_calendar,
            currency_data,
            NMD_adjustments,
            repayment_schedule,
            market_data,
            vix_data,
            cf_analysis_id,
            cashflow_uploaded_data,
            underlying_position_data,
            custom_daycount_conventions,
            dpd_ruleset,
            overdue_bucketing_data,
            dpd_schedule,
            product_holiday_code,
            request,
            curve_data,
            credit_spread_data,
        ),
    ) as pool:
        logging.warning("Starting pool.map on position data")
        results = pool.map(worker_func, pos_data)
        logging.warning(f"Finished pool.map, number of results: {len(results)}")

    successes = []
    failures = []

    for res in results:
        if res[0] == "success":
            successes.append(res[1])  # res[1] contains the result from func()
        else:
            # Now, res[1] is unique_reference_id and res[2] is the error message
            failures.append({"unique_reference_id": res[1], "error": res[2]})
            logging.warning(
                f"Recorded failure for Unique_Reference_Id {res[1]}: {res[2]}"
            )

    logging.warning("Completed processing results: successes and failures recorded")

    # Process successful results if any
    if successes:
        try:
            retLst, cashflow_model_results, measures_outputs = zip(*successes)
            retLst = list(retLst)
            cashflow_model_results = list(cashflow_model_results)
            measures_outputs = list(measures_outputs)
            logging.warning(
                f"Successfully unpacked results: {len(retLst)} processed positions"
            )
        except Exception as e:
            logging.warning(f"Error unpacking success results: {e}")
            retLst, cashflow_model_results, measures_outputs = [], [], []
    else:
        logging.warning("No successful positions processed")
        retLst, cashflow_model_results, measures_outputs = [], [], []

    # Concatenate cashflow model results into one DataFrame
    cashflow_output = pd.DataFrame()
    for df in cashflow_model_results:
        if isinstance(df, pd.DataFrame) and not df.empty:
            cashflow_output = pd.concat([cashflow_output, df], ignore_index=True)
    logging.warning(
        f"Consolidated cashflow model results; total rows: {len(cashflow_output)}"
    )

    # Concatenate measures outputs into one DataFrame
    measures_output = pd.DataFrame()
    for df in measures_outputs:
        if isinstance(df, pd.DataFrame) and not df.empty:
            measures_output = pd.concat([measures_output, df], ignore_index=True)
    logging.warning(
        f"Consolidated measures outputs; total rows: {len(measures_output)}"
    )

    final_output = pd.DataFrame(retLst)
    logging.warning(f"Final output DataFrame created with {len(final_output)} rows")

    # Save failures to a parquet file if any failures occurred
    if failures:
        failed_df = pd.DataFrame(failures)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"/opt/revolutio/Platform_Configs/failed_pos_{timestamp}.parquet"
        try:
            failed_df.to_parquet(filename)
            logging.warning(
                f"Failure details saved to {filename} with {len(failed_df)} records"
            )
        except Exception as e:
            logging.warning(f"Could not save failure details to parquet: {e}")

    logging.warning("Returning from applyParallel")
    return final_output, cashflow_output, measures_output


def day_count(day_count_list):
    day_count_dict = {
        "ACT/365": 365,
        "ACT/360": 360,
        "ACT/252": 252,
        "ACT/ACT": 365.25
    }
    return [day_count_dict.get(day_count, 365.25) for day_count in day_count_list]


def central_curve_processing(
    valuation_date=None,
    pos_data=None,
    column_index_dict=None,
    curve_repo_data=None,
    curve_components_data=None,
    cs_curve_repo_data=None,
    cs_curve_components_data=None,
    mtm_data=None,
):

    # Replace empty strings with None in the specified columns
    columns_to_replace = [
        'credit_spread_curve',
        'discounting_curve',
        'forward_benchmark_curve',
        'discounting_curve_secondary_currency',
    ]
    for column in columns_to_replace:
        pos_data[column].replace('', None, inplace=True)
        pos_data[column] = pos_data[column].astype(str)

    curve_component_transformation_vect = np.vectorize(curve_component_transformation)
    curve_data = pd.DataFrame()
    credit_spread_data = pd.DataFrame()

    unique_curve_list = []
    unique_curve_list += pos_data['credit_spread_curve'].unique().tolist()
    unique_curve_list += pos_data['discounting_curve'].unique().tolist()
    unique_curve_list += pos_data['forward_benchmark_curve'].unique().tolist()
    unique_curve_list += pos_data['discounting_curve_secondary_currency'].unique().tolist()

    unique_curve_list = list(set(unique_curve_list))
    unexpected_curve_name = ['nan', np.nan, " ", "", np.nan, 0,"None"]
    filtered_list = [item for item in unique_curve_list if item is not None and item not in unexpected_curve_name]
    unique_curve_list = filtered_list

    new_curve = []
    total_curve = np.append(curve_repo_data['curve_name'], cs_curve_repo_data['curve_name'])

    new_curve.extend([curve_name for curve_name in unique_curve_list if curve_name not in total_curve])
    new_curve = [new_curve_i for new_curve_i in new_curve if new_curve_i is not None]

    if len(new_curve) > 0:
        raise Exception(f"Curve not found in Curve Repository: {new_curve}")

    curve_repo_data = curve_repo_data[curve_repo_data['curve_name'].isin(unique_curve_list)]
    cs_curve_repo_data = cs_curve_repo_data[cs_curve_repo_data['curve_name'].isin(unique_curve_list)]

    # Process curve_repo_data and curve_components_data
    if len(curve_repo_data) > 0:
        curve_component_transformation_result = curve_component_transformation_vect(
            curve_repo_data.to_dict("records")
        )
        curve_data = pd.concat(curve_component_transformation_result, ignore_index=True)
        del curve_component_transformation_result

        curve_data = curve_data.merge(
            curve_components_data, left_on="curve_components", right_on="id", how="left"
        ).drop(columns=["curve_components", "id"])

        day_count_list = day_count(curve_data['day_count'])
        curve_data = curve_data.drop(columns='day_count')
        curve_data['day_count_value'] = day_count_list

        curve_data["tenor"] = curve_data.apply(
            lambda row: row["tenor_value"] / row["day_count_value"]
            if row["tenor_unit"] == "D"
            else (row["tenor_value"] / 12 if row["tenor_unit"] == "M" else row["tenor_value"]),
            axis=1
        )

        # Merge between Market Data and Curve Data to achieve mapping between Security Identifier & Rates
        curve_components_check = curve_data['curve_component'].unique().tolist()
        security_identifiers_check = mtm_data['security_identifier'].unique().tolist()
        missing_components = [
            component for component in curve_components_check if component not in security_identifiers_check
        ]
        if len(missing_components) > 0:
            raise Exception(f"Interest Rate Curve Market data missing: {missing_components}")

        curve_data = (
            curve_data.merge(
                mtm_data.loc[
                    mtm_data["extract_date"] == pd.to_datetime(valuation_date, dayfirst=True),
                    ["security_identifier", "quoted_price"],
                ],
                left_on="curve_component",
                right_on="security_identifier",
                how="left",
            )
            .drop(columns=["security_identifier"])
            .rename(columns={"quoted_price": "rate"})
        )
        curve_data.sort_values(by=["curve_name", "tenor"], inplace=True)

    # Process cs_curve_repo_data and cs_curve_components_data
    if len(cs_curve_repo_data) > 0:
        cs_curve_component_transformation_result = curve_component_transformation_vect(
            cs_curve_repo_data.to_dict("records")
        )
        cs_curve_data = pd.concat(cs_curve_component_transformation_result, ignore_index=True)
        del cs_curve_component_transformation_result

        cs_curve_data = cs_curve_data.merge(
            cs_curve_components_data, left_on="curve_components", right_on="id", how="left"
        ).drop(columns=["curve_components", "id"])

        day_count_list = day_count(cs_curve_data['day_count'])
        cs_curve_data = cs_curve_data.drop(columns='day_count')
        cs_curve_data['day_count_value'] = day_count_list

        cs_curve_data["tenor"] = cs_curve_data.apply(
            lambda row: row["tenor_value"] / row["day_count_value"]
            if row["tenor_unit"] == "D"
            else (row["tenor_value"] / 12 if row["tenor_unit"] == "M" else row["tenor_value"]),
            axis=1
        )

        cs_curve_components_data_check = cs_curve_data['curve_component'].unique().tolist()
        security_identifiers_check = mtm_data['security_identifier'].unique().tolist()
        missing_components = [
            component for component in cs_curve_components_data_check if component not in security_identifiers_check
        ]
        if len(missing_components) > 0:
            raise Exception(f"Credit Spread Curve Market data missing: {missing_components}")

        # Merge the cs_curve_repo with mtm_data
        credit_spread_data = (
            cs_curve_data.merge(
                mtm_data.loc[
                    mtm_data["extract_date"] == pd.to_datetime(valuation_date, dayfirst=True),
                    ["security_identifier", "quoted_price"],
                ],
                left_on="curve_component",
                right_on="security_identifier",
                how="left",
            )
            .drop(columns=["security_identifier"])
            .rename(columns={"quoted_price": "spread_value", "curve_name": "credit_spread_curve_name"})
        )
        del cs_curve_data
        credit_spread_data.sort_values(by=["credit_spread_curve_name", "tenor"], inplace=True)
        credit_spread_data = credit_spread_data.drop(columns=["tenor"]).rename(columns={"tenor_value": "tenor"})

    return curve_data, credit_spread_data


def final_valuation_fn(config_dict, data=None):
    
    request_user = request.user.username
    valuation_date = config_dict["inputs"]["Valuation_Date"]["val_date"]
    cf_analysis_id = config_dict["inputs"]["CF_Analysis_Id"]["cf_analysis_id"]

    if not cf_analysis_id:  # This checks for both None and empty string
        raise ValueError("Please reconfigure cf_analysis_id in Portfolio valuation element")
    
    val_date_filtered = data["positions_table"].copy()

    if not valuation_date:  # This checks for both None and empty string
        raise ValueError("Please reconfigure valuation date in Portfolio valuation element")
    
    val_date_filtered = val_date_filtered[val_date_filtered["reporting_date"] == pd.to_datetime(valuation_date) ]
    val_date_filtered = val_date_filtered.drop_duplicates(subset=['position_id'])
    val_date_filtered = val_date_filtered.sort_values(by='position_id')
    
    if len(val_date_filtered) < 1:
        raise ValueError(f"No position data found for {valuation_date} . Please try again.")
    
    
    NMD_adjustments = data["nmd_data"].copy()
    product_data = data["product_data"].copy()
    dpd_ruleset = data["dpd_data"].copy()
    overdue_bucketing_data = data["overdue_data"].copy()
    dpd_schedule = data["dpd_schedule"].copy()
    mtm_data = data["market_data"].copy()
    repayment_schedule = data["repayment_data"].copy()
    repayment_schedule = repayment_schedule.groupby(['payment_date','position_id'])['payment_amount'].sum().reset_index()
    product_model_mapper = data["product_model_mapper_table"].copy()
    cashflow_uploaded_data = data["cashflow_data_uploaded"].copy()

    
    #import as fast excute require consitence uploaded data are error prone
    date_columns = [col for col in cashflow_uploaded_data.columns if '_date' in col]   
    for date in date_columns:
        cashflow_uploaded_data[date] = pd.to_datetime(cashflow_uploaded_data[date]).dt.date

    float_columns = ['cashflow', 'time_to_maturity', 'discount_factor', 'present_value']
    existing_float_columns = [col for col in float_columns if col in cashflow_uploaded_data.columns]
    for col in existing_float_columns:
        cashflow_uploaded_data[col] = pd.to_numeric(cashflow_uploaded_data[col], errors='coerce').fillna(0.0)
    
    

    # if data["nmd_data"] is not None:
    #     NMD_adjustments = data["nmd_data"].copy()
    # else:
    #     NMD_adjustments = pd.DataFrame()
    # if data["product_data"] is not None:
    #     product_data = data["product_data"].copy()
    # else:
    #     product_data = pd.DataFrame()
    # if data["dpd_data"] is not None:
    #     dpd_ruleset = data["dpd_data"].copy()
    # else:
    #     dpd_ruleset = pd.DataFrame()
    # if data["overdue_data"] is not None:
    #     overdue_bucketing_data = data["overdue_data"].copy()
    # else:
    #     overdue_bucketing_data = pd.DataFrame()
    # if data["dpd_schedule"] is not None:
    #     dpd_schedule = data["dpd_schedule"].copy()
    # else:
    #     dpd_schedule = pd.DataFrame()

    # if data["cashflow_data_uploaded"] is not None:
    #     cashflow_uploaded_data = data["cashflow_data_uploaded"].copy()
        
    #     #import as fast excute require consitence uploaded data are error prone
    #     date_columns = [col for col in cashflow_uploaded_data.columns if '_date' in col]   
    #     for date in date_columns:
    #         cashflow_uploaded_data[date] = pd.to_datetime(cashflow_uploaded_data[date]).dt.date

    #     float_columns = ['cashflow', 'time_to_maturity', 'discount_factor', 'present_value']
    #     existing_float_columns = [col for col in float_columns if col in cashflow_uploaded_data.columns]
    #     for col in existing_float_columns:
    #         cashflow_uploaded_data[col] = pd.to_numeric(cashflow_uploaded_data[col], errors='coerce').fillna(0.0)
        
    # else:
    #     cashflow_uploaded_data = pd.DataFrame()

    # if data["market_data"] is not None and len(data["market_data"])>0:
    #     mtm_data = data["market_data"].copy()
    # else:
    #     mtm_data = pd.DataFrame(columns=['extract_date','security_identifier','asset_class','quoted_price','yield','volatility'])
    
    # if data["repayment_data"] is not None:
    #     repayment_schedule = data["repayment_data"].copy()
    #     repayment_schedule = repayment_schedule.groupby(['payment_date','position_id'])['payment_amount'].sum().reset_index()
    # else:
    #     repayment_schedule = pd.DataFrame()
        
    # if data["product_model_mapper_table"] is not None:
    #     product_model_mapper = data["product_model_mapper_table"].copy()
    # else:
    #     product_model_mapper = pd.DataFrame()

    if (
        "hierarchy_name" in val_date_filtered.columns
        and "product_variant_name" not in val_date_filtered.columns
    ):
        val_date_filtered.rename(columns={"hierarchy_name": "product_variant_name"}, inplace=True)
    data = None
    del data

    product_model_mapper = product_model_mapper.set_index("product_variant_name").to_dict()["model_code"]

    val_date_filtered["model_code"] = val_date_filtered["product_variant_name"].replace(product_model_mapper)

    holiday_code_generation = np.vectorize(holiday_code_generator)

    # data_path_dict for replacement of read_data_func
    data_directory = "Read Data Func Data"
    empty_data_directory = "Read Data Func Sample Data"
    data_path_dict = {
        "weekday_data" : "week_definition.csv", 
        "curve_repo_data" : "ir_curve_repository.csv", 
        "curve_components_data" : "ir_curve_components.csv", 
        "vol_repo_data" : "volatility_surface_repository.csv", 
        "vol_components_data" : "volatility_surface_components.csv", 
        "cs_curve_repo_data" : "cs_curve_repository.csv", 
        "cs_curve_components_data" : "cs_curve_components.csv", 
        "custom_daycount_conventions" : "custom_daycount_conventions.csv", 
        "holiday_calendar" : "Holiday_Calendar_Repository.csv", 
        "currency_data" : "CurrencyMaster.csv", 
        "vix_data" : "", 
    }
    read_data_func_data = {}
    for table, path in data_path_dict.items():
        try:
            full_path = os.join(data_directory, path)
            read_data_func_data[table] = pd.read_csv(full_path)
        except FileNotFoundError:
            full_path = os.join(empty_data_directory, path)
            read_data_func_data[table] = pd.read_csv(full_path)

    weekday_data = read_data_func_data['weekday_data']
    curve_repo_data = read_data_func_data['curve_repo_data']
    curve_components_data = read_data_func_data['curve_components_data']
    vol_repo_data = read_data_func_data['vol_repo_data']
    vol_components_data = read_data_func_data['vol_components_data']
    cs_curve_repo_data = read_data_func_data['cs_curve_repo_data']
    cs_curve_components_data = read_data_func_data['cs_curve_components_data']
    custom_daycount_conventions = read_data_func_data['custom_daycount_conventions']
    holiday_calendar = read_data_func_data['holiday_calendar']
    currency_data = read_data_func_data['currency_data']
    vix_data = read_data_func_data['vix_data']

    # weekday_data = json.dumps(
    #     read_data_func(
    #         request,
    #         {
    #             "inputs": {
    #                 "Data_source": "Database",
    #                 "Table": "week_definition",
    #                 "Columns": ["id", "day"],
    #             },
    #             "condition": [],
    #         },
    #     ).to_dict("list")
    # )

    # curve_repo_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "ir_curve_repository",
    #             "Columns": [
    #                 "configuration_date",
    #                 "curve_name",
    #                 "curve_components",
    #                 "interpolation_algorithm",
    #                 "extrapolation_algorithm",
    #                 "compounding_frequency_output",
    #                 'day_count',
    #             ],
    #         },
    #         "condition": [
    #             {
    #                 "column_name": "configuration_date",
    #                 "condition": "Smaller than equal to",
    #                 "input_value": str(valuation_date),
    #                 "and_or": "",
    #             },
    #         ],
    #     },
    # )


    # curve_components_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "ir_curve_components",
    #             "Columns": ["id", "curve_component", "tenor_value", "tenor_unit"],
    #         },
    #         "condition": [],
    #     },
    # )
    if any(
        model in ["M027", "M014", "M015", "M016", "M017", "M040", "M041", "M042", "M043", "M044"]
        for model in val_date_filtered["model_code"].tolist()
    ):
        # vol_repo_data = read_data_func(
        #     request,
        #     {
        #         "inputs": {
        #             "Data_source": "Database",
        #             "Table": "volatility_surface_repository",
        #             "Columns": [
        #                 "configuration_date",
        #                 "vol_surface_name",
        #                 "vol_surface_components",
        #                 "interpolation_smile",
        #                 "interpolation_tenor",
        #                 "extrapolation_smile",
        #                 "extrapolation_tenor",
        #                 "tenor_interpolation_parameter",
        #                 "smile_interpolation_parameter",
        #                 "asset_class",
        #             ],
        #         },
        #         "condition": [
        #             {
        #                 "column_name": "configuration_date",
        #                 "condition": "Smaller than equal to",
        #                 "input_value": str(valuation_date),
        #                 "and_or": "",
        #             },
        #         ],
        #     },
        # )

        vol_repo_data = vol_repo_data.loc[vol_repo_data['configuration_date'] <= str(valuation_date)].reset_index()
        vol_repo_data = vol_repo_data.sort_values("configuration_date", ascending=False).drop_duplicates(
            subset=["vol_surface_name"]
        )
        # vol_components_data = read_data_func(
        #     request,
        #     {
        #         "inputs": {
        #             "Data_source": "Database",
        #             "Table": "volatility_surface_components",
        #             "Columns": ["id", "surface_component", "tenor_value", "tenor_unit", "delta"],
        #         },
        #         "condition": [],
        #     },
        # )
    # else:
    #     vol_repo_data = pd.DataFrame()
    #     vol_components_data = pd.DataFrame()

    # cs_curve_repo_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "cs_curve_repository",
    #             "Columns": [
    #                 "configuration_date",
    #                 "curve_name",
    #                 "curve_components",
    #                 "interpolation_algorithm",
    #                 "extrapolation_algorithm",
    #                 'day_count',
    #             ],
    #         },
    #         "condition": [
    #             {
    #                 "column_name": "configuration_date",
    #                 "condition": "Smaller than equal to",
    #                 "input_value": str(valuation_date),
    #                 "and_or": "",
    #             },
    #         ],
    #     },
    # )
    
    # cs_curve_components_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "cs_curve_components",
    #             "Columns": ["id", "curve_component", "tenor_value", "tenor_unit"],
    #         },
    #         "condition": [],
    #     },
    # )
    # custom_daycount_conventions = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "custom_daycount_conventions",
    #             "Columns": [
    #                 "convention_name",
    #                 "numerator",
    #                 "denominator",
    #                 "numerator_adjustment",
    #                 "denominator_adjustment",
    #             ],
    #         },
    #         "condition": [],
    #     },
    # )

    
    # holiday_calendar = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "Holiday_Calendar_Repository",
    #             "Columns": ["holiday_calendar", "holiday_date"],
    #         },
    #         "condition": [],
    #     },
    # )
    # currency_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "CurrencyMaster",
    #             "Columns": ["currency_code", "default_holiday_calendar"],
    #         },
    #         "condition": [],
    #     },
    # )

    # vix_data = read_data_func(
    #     request,
    #     {
    #         "inputs": {
    #             "Data_source": "Database",
    #             "Table": "vix",
    #             "Columns": ["extract_date", "vix"],
    #         },
    #         "condition": [
    #             {
    #                 "column_name": "extract_date",
    #                 "condition": "Equal to",
    #                 "input_value": str(valuation_date),
    #                 "and_or": "",
    #             },
    #         ],
    #     },
    # )
    
    product_holiday_code = pd.concat(
        holiday_code_generation(product_data.fillna("None").to_dict("records"), weekday_data),
        ignore_index=True,
    )
    curve_repo_data = curve_repo_data.loc[curve_repo_data['configuration_date'] <= str(valuation_date)].reset_index()
    curve_repo_data = curve_repo_data.sort_values("configuration_date", ascending=False).drop_duplicates(
        subset=["curve_name"]
    )
    cs_curve_repo_data = cs_curve_repo_data.loc[cs_curve_repo_data['configuration_date'] <= str(valuation_date)].reset_index()
    cs_curve_repo_data = cs_curve_repo_data.sort_values(
        "configuration_date", ascending=False
    ).drop_duplicates(subset=["curve_name"])

    if "strike_price" in val_date_filtered.columns and "put_call_type" in val_date_filtered.columns:
        underlying_position_data = val_date_filtered.loc[
            :,
            [
                "underlying_position_id",
                "unique_reference_id",
                "maturity_date",
                "reporting_date",
                "strike_price",
                "put_call_type",
                "product_variant_name",
            ],
        ]
    else:
        underlying_position_data = pd.DataFrame(
            columns=[
                "underlying_position_id",
                "unique_reference_id",
                "maturity_date",
                "reporting_date",
                "strike_price",
                "put_call_type",
                "product_variant_name",
            ]
        )

    position_security_id = (
        val_date_filtered[val_date_filtered["unique_reference_id"].notna()]["unique_reference_id"]
        .unique()
        .tolist()
    )
    if (val_date_filtered["underlying_position_id"].str.contains(",")).any():
        a = val_date_filtered["underlying_position_id"].unique().tolist()
        b = set()
        for i in range(len(a)):
            temp = a[i].split(",")
            for j in temp:
                b.add(j)
        c = list(b)
        position_security_id += c
    else:
        position_security_id += (
            val_date_filtered[val_date_filtered["underlying_position_id"].notna()]["underlying_position_id"]
            .unique()
            .tolist()
        )
    position_security_id += curve_components_data["curve_component"].unique().tolist()
    position_security_id += (
        underlying_position_data[underlying_position_data["underlying_position_id"].notna()][
            "underlying_position_id"
        ]
        .unique()
        .tolist()
    )
    position_security_id += cs_curve_components_data["curve_component"].unique().tolist()
    mtm_data = pd.concat(
        (
            mtm_data.loc[mtm_data["security_identifier"].isin(position_security_id)],
            mtm_data.loc[mtm_data["asset_class"] == "FX"],
        ),
        ignore_index=True,
    )
    mtm_data["extract_date"] = mtm_data["extract_date"].astype('datetime64[ns]')
    vix_data = vix_data.loc[vix_data['extract_date'] == str(valuation_date)]

    table_cols = val_date_filtered.columns
    index_list = []

    def col_index_func(table_cols):
        index_list.append(val_date_filtered.columns.get_loc(table_cols))

    np.vectorize(col_index_func)(table_cols)
    index_list.pop(0)

    column_index_dict = dict(zip(table_cols, index_list))

    final_output_main = pd.DataFrame()
    run_id = "run_" + str(random.random()).replace(".","")

    import glob
    paths = {
        # 'cashflow': f'{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Cashflow/',
        # 'measures': f'{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Measures/',
        'Information': f'{DISKSTORE_PATH}/Cashflow_Engine_Outputs/Information/',
    }

    total_files_removed = 0
    for key, path in paths.items():
        parquet_files = glob.glob(os.path.join(path, '*.parquet'))
        file_count = len(parquet_files)
        for file in parquet_files:
            os.remove(file)

        logging.warning(f'Total number of files removed at {key} are: {file_count}')
        total_files_removed += file_count


    logging.warning(f'Total number of files removed across all paths: {total_files_removed}')

    total_size_bytes = val_date_filtered.memory_usage(deep=True).sum()

    def calculate_chunk_size(val_date_filtered):
        def get_l3_cache_size():
            try:
                if platform.system() == 'Linux':
                
                    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
                    for line in result.stdout.split('\n'):
                        if 'L3 cache' in line:
                
                            match = re.search(r'(\d+)', line)
                            if match:
                                return int(match.group(1)) * 1024 * 1024  
                elif platform.system() == 'Windows':
                    
                    result = subprocess.run(['wmic', 'cpu', 'get', 'L3CacheSize'], stdout=subprocess.PIPE, text=True)
                    for line in result.stdout.split('\n'):
                        if line.strip().isdigit():
                    
                            return int(line.strip()) * 1024  
            except Exception as e:
                
                return None

        total_size_bytes = val_date_filtered.memory_usage(deep=True).sum()
        number_of_positions = len(val_date_filtered)

        l3_cache_size = get_l3_cache_size()
        if l3_cache_size:
            ideal_chunk_size = l3_cache_size // (total_size_bytes // number_of_positions)
            return ideal_chunk_size * 10
        else:
            return 1000

    chunk_size = calculate_chunk_size(val_date_filtered)
    logging.warning(f'Chunk_size {chunk_size}')
    logging.warning(f'Chunk_size {chunk_size}')
    logging.warning(f'Chunk_size {chunk_size}')


    aggregation_mapping = {
        "M048": {
            "target_model_code": "M065",
            "amount_col": "outstanding_amount"
        },
        "M049": {
            "target_model_code": "M065",
            "amount_col": "outstanding_amount"
        },
        "M050": {
            "target_model_code": "M075",
            "amount_col": "unutilized_limit_amount"
        },
    }

    group_columns = ["npa_flag"]  # can add more column like npa_category , nmd_cashflow_freqency 

    def sanitize_group_key(group_key):
        """
        Convert a group key (which might be a tuple or a single value) into a
        string that contains only alphanumeric characters and underscores.
        Booleans are converted to "1" (True) and "0" (False).
        """
        # Ensure group_key is a tuple for uniform processing.
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        
        safe_parts = []
        for val in group_key:
            if isinstance(val, bool):
                # Convert booleans to "1" or "0"
                safe_val = "1" if val else "0"
            else:
                # Remove any character that is not alphanumeric or underscore.
                safe_val = re.sub(r'[^A-Za-z0-9_]', '', str(val))
            safe_parts.append(safe_val)

        return "_".join(safe_parts)


    val_date_filtered_deposits_aggregate_model = val_date_filtered.loc[
        val_date_filtered["model_code"].isin(aggregation_mapping.keys())
    ]

    if len(val_date_filtered_deposits_aggregate_model) > 0:
        for source_model_code, agg_info in aggregation_mapping.items():
            sub_df = val_date_filtered_deposits_aggregate_model.loc[
                val_date_filtered_deposits_aggregate_model["model_code"] == source_model_code
            ]

            if len(sub_df) == 0:
                # No data for this particular model_code
                continue

            amount_col = agg_info["amount_col"]
            target_model_code = agg_info["target_model_code"]
            
            # ---------------------------------------------------
            # Group sub_df by any columns you need to handle separately
            # For example: by npa_flag, npa_category, etc.
            # This way, if "npa_flag" = 1 and "npa_flag"=0, you get separate aggregated rows.
            # ---------------------------------------------------
            grouped_aggregations = []
            for group_key, group_data in sub_df.groupby(group_columns):
                representative_row = group_data.head(1).copy()

                # Weighted average for base_rate
                weighted_base_rate = (
                    (group_data[amount_col] * group_data["base_rate"]).sum()
                    / group_data[amount_col].sum()
                )

                # Weighted average for fixed_spread
                weighted_fixed_spread = (
                    (group_data[amount_col] * group_data["fixed_spread"]).sum()
                    / group_data[amount_col].sum()
                )

                # Sum of accrued_interest
                accrued_interest_sum = group_data["accrued_interest"].sum()

                product_variant_name = representative_row["product_variant_name"].iloc[0].replace(" ", "_")
                group_suffix = sanitize_group_key(group_key)
                new_position_id = f"{product_variant_name}_Aggregated_{group_suffix}"


                # Modify the representative row to become the "aggregated" position
                representative_row = representative_row.assign(
                    **{
                        "outstanding_amount": group_data[amount_col].sum()
                        if amount_col == "outstanding_amount"
                        else group_data["outstanding_amount"].sum(),  
                        "unutilized_limit_amount": group_data[amount_col].sum()
                        if amount_col == "unutilized_limit_amount"
                        else group_data["unutilized_limit_amount"].sum(),  
                        "position_id": new_position_id,
                        "unique_reference_id": new_position_id,
                        "model_code": target_model_code,
                        "base_rate": weighted_base_rate,
                        "fixed_spread": weighted_fixed_spread,
                        "accrued_interest": accrued_interest_sum,
                    }
                )

                grouped_aggregations.append(representative_row)

            if len(grouped_aggregations) > 0:
                # Concatenate all new aggregated rows
                aggregated_positions_df = pd.concat(grouped_aggregations, ignore_index=True)

                # Remove old rows of the source model_code
                val_date_filtered = val_date_filtered.loc[
                    ~(val_date_filtered["model_code"] == source_model_code)
                ]

                # Append the new aggregated rows
                val_date_filtered = pd.concat(
                    [val_date_filtered, aggregated_positions_df], ignore_index=True
                )

    error_pos_data = pd.DataFrame()
    if isinstance(val_date_filtered, pd.DataFrame):
        val_date_filtered,error_pos_data  =  preprocess_position_data(val_date_filtered,column_index_dict,valuation_date,cashflow_uploaded_data)

    curve_data = pd.DataFrame()
    credit_spread_data = pd.DataFrame()

    # check from input, output_choice's to check for valuation &&  use_fixed_or_float_flag  for floating position's 
    # check all the unique curve name  ::  3 column , forward_curve , dicounting curve , primary , secondry , credit spread curve 

    if (config_dict["inputs"]["Output_choice"]["Valuation"] == "Yes") or val_date_filtered['fixed_or_float_flag'].str.contains("Float", case=False).any():
        
        curve_data,credit_spread_data = central_curve_processing(
                    valuation_date,
                    val_date_filtered, 
                    column_index_dict,
                    curve_repo_data,
                    curve_components_data,
                    cs_curve_repo_data ,
                    cs_curve_components_data,
                    mtm_data)


    if len(val_date_filtered)>0:
        val_date_filtered_array = np.array(val_date_filtered)
    else:
        val_date_filtered_array = []
    
    val_date_filtered = []
    del val_date_filtered

    func = Valuation_Models.Value_extraction_pf


    def process_product_variant(
        product_variant_name,
        legal_entity_name,
        val_date_filtered_array,
        valuation_date,
        run_id,
        config_dict,
        column_index_dict,
        vol_repo_data,
        vol_components_data,
        holiday_calendar,
        currency_data,
        NMD_adjustments,
        repayment_schedule,
        func,
        vix_data,
        cf_analysis_id,
        cashflow_uploaded_data,
        underlying_position_data,
        custom_daycount_conventions,
        dpd_ruleset,
        overdue_bucketing_data,
        dpd_schedule,
        product_holiday_code,
        request,
        mtm_data,
        curve_data,
        credit_spread_data,
        chunk_size,
        DISKSTORE_PATH,
        final_output_main,
        completion_percent
    ):

        pos_id_col = column_index_dict["position_id"]
        unique_ref_id_col = column_index_dict["unique_reference_id"]
        product_variant_col = column_index_dict["product_variant_name"]
        legal_entity_col = column_index_dict["legal_entity"]

        subset_mask = (
            (val_date_filtered_array[:, legal_entity_col] == legal_entity_name) &
            (val_date_filtered_array[:, product_variant_col] == product_variant_name)
        )
        variant_filtered = val_date_filtered_array[subset_mask]

        if len(variant_filtered) == 0:
            return final_output_main


        cashflow_dir = Path(f"{DISKSTORE_PATH}Cashflow_Engine_Outputs/Cashflow")
        measures_dir = Path(f"{DISKSTORE_PATH}Cashflow_Engine_Outputs/Measures")

        def remove_older_date_files(directory, file_prefix):
            """
            Removes any file in 'directory' matching the given prefix that has a date < current valuation_date. 
            Example naming pattern to parse:
            cashflow_output_{legal_entity}_{valuation_date}_{product_variant}_{chunk_id}_{run_id}.parquet
            => we expect:
            [0] = "cashflow" 
            [1] = "output"
            [2] = legal_entity 
            [3] = valuation_date
            [4] = product_variant
            [5] = chunk_id
            [6] = run_id
            """
            all_files = directory.glob(f"{file_prefix}_{legal_entity_name}_*.parquet")
            current_val_date = datetime.strptime(valuation_date, "%Y-%m-%d")

            for f in all_files:
                parts = f.stem.split('_')  # "cashflow_output_ABC_2025-01-21_ProductX_1_1001"
                if len(parts) < 5:
                    continue  # Not a valid file name format, skip

                try:
                    file_date_str = parts[3]  # Adjust pattern is different
                    file_date_obj = datetime.strptime(file_date_str, "%Y-%m-%d")
                    if file_date_obj < current_val_date:
                        os.remove(f)
                except Exception:
                    # If parsing fails, skip or handle as needed
                    pass

        # Remove older-date files for cashflow + measures for (legal_entity_name, product_variant_name)
        remove_older_date_files(cashflow_dir, "cashflow_output")
        remove_older_date_files(measures_dir, "measures_output")


        existing_cashflow_files = sorted(
            cashflow_dir.glob(f'*_{legal_entity_name}_{valuation_date}_{product_variant_name}_*.parquet')
        )
        existing_measures_files = sorted(
            measures_dir.glob(f'*_{legal_entity_name}_{valuation_date}_{product_variant_name}_*.parquet')
        )

        new_unique_refs = set(variant_filtered[:, pos_id_col])

        def clean_existing_files(file_list, ref_id_col="position_id"):
            # Start from the last file to leverage temporal locality
            for f in reversed(file_list):
                if not new_unique_refs:
                    break  

                try:
                    df = pd.read_parquet(f)
                except Exception as e:
                    # If we can't read this file as Parquet, log a warning and remove it.
                    os.remove(f)
                    continue
                
                mask = df[ref_id_col].isin(new_unique_refs)
                removed_ids = set(df.loc[mask, ref_id_col].unique())

                df = df[~mask]  # Filter out rows with new_unique_refs

                os.remove(f)  # Remove the file
                if not df.empty:
                    pq.write_table(pa.Table.from_pandas(df), f)

                if removed_ids:
                    new_unique_refs.difference_update(removed_ids)

        # Clean existing partial matches for current date
        if existing_cashflow_files:
            clean_existing_files(existing_cashflow_files, ref_id_col="position_id")
        if existing_measures_files:
            clean_existing_files(existing_measures_files, ref_id_col="position_id")


        num_splits = max(1, int(np.ceil(len(variant_filtered) / chunk_size)))
        completed_so_far = 0

        def get_max_identifier(file_list):
            max_id = 0
            for f in file_list:
                parts = f.stem.split('_')
                try:
                    # chunk_id might be 2nd from last if pattern is *_{chunk_id}_{run_id}.parquet
                    file_id = int(parts[-2])
                    max_id = max(max_id, file_id)
                except:
                    pass
            return max_id

        max_cashflow_id = get_max_identifier(existing_cashflow_files)
        max_measures_id = get_max_identifier(existing_measures_files)

        cashflow_identifier = max_cashflow_id + 1
        measures_identifier = max_measures_id + 1

        for chunk_index, chunk_pos_data in enumerate(
            np.array_split(variant_filtered, num_splits), start=1
        ):
            completed_so_far += len(chunk_pos_data)
            completion_percent(
                completed_so_far, variant_filtered, chunk_index, num_splits, product_variant_name
            )

            # Filter cashflow_uploaded_data if needed
            if len(cashflow_uploaded_data) > 0:
                chunk_pos_ids = chunk_pos_data[:, pos_id_col]
                cashflow_uploaded_data_filtered = cashflow_uploaded_data.loc[
                    cashflow_uploaded_data["position_id"].isin(chunk_pos_ids)
                ]
            else:
                cashflow_uploaded_data_filtered = pd.DataFrame()

            # (Your existing parallel application logic)
            final_output, cashflow_output, measures_output = applyParallel(
                config_dict,
                column_index_dict,
                chunk_pos_data,
                vol_repo_data,
                vol_components_data,
                holiday_calendar,
                currency_data,
                NMD_adjustments,
                repayment_schedule,
                func,
                vix_data,
                cf_analysis_id,
                cashflow_uploaded_data_filtered,
                underlying_position_data,
                custom_daycount_conventions,
                dpd_ruleset,
                overdue_bucketing_data,
                dpd_schedule,
                product_holiday_code,
                request,
                mtm_data,
                curve_data,
                credit_spread_data,
            )
            del cashflow_uploaded_data_filtered

            if len(cashflow_output) > 0:
                if "cf_analysis_id" not in cashflow_output.columns:
                    cashflow_columns = ["cf_analysis_id"] + cashflow_output.columns.tolist()
                else:
                    # reorder so that 'cf_analysis_id' is first
                    cf_cols = cashflow_output.drop(columns=["cf_analysis_id"]).columns.tolist()
                    cashflow_columns = ["cf_analysis_id"] + cf_cols

                cashflow_output["cf_analysis_id"] = cf_analysis_id
                cashflow_output = cashflow_output.loc[:, cashflow_columns]

                if "cashflow" in cashflow_output.columns:
                    cashflow_output = cashflow_output.loc[cashflow_output["cashflow"].notnull()]

                if not cashflow_output.empty:
                    cf_table = pa.Table.from_pandas(cashflow_output)
                    output_path = (
                        f"{DISKSTORE_PATH}Cashflow_Engine_Outputs/Cashflow/"
                        f"cashflow_output_{legal_entity_name}_{valuation_date}_"
                        f"{product_variant_name}_{cashflow_identifier}_{run_id}.parquet"
                    )
                    pq.write_table(cf_table, output_path)
                    cashflow_identifier += 1

            if len(measures_output) > 0:
                measures_output["cf_analysis_id"] = cf_analysis_id
                # Drop null measure_values if needed
                if "measure_value" in measures_output.columns:
                    measures_output = measures_output.loc[measures_output["measure_value"].notnull()]

                if not measures_output.empty:
                    ms_table = pa.Table.from_pandas(measures_output)
                    output_path = (
                        f"{DISKSTORE_PATH}Cashflow_Engine_Outputs/Measures/"
                        f"measures_output_{legal_entity_name}_{valuation_date}_"
                        f"{product_variant_name}_{measures_identifier}_{run_id}.parquet"
                    )
                    pq.write_table(ms_table, output_path)
                    measures_identifier += 1

            if len(final_output_main) < 100:
                final_output_main = pd.concat([final_output_main, final_output], ignore_index=True)

            del final_output
            del cashflow_output
            del measures_output

        return final_output_main


    legal_entity_col = column_index_dict["legal_entity"]
    product_variant_col = column_index_dict["product_variant_name"]

    all_legal_entities = np.unique(val_date_filtered_array[:, legal_entity_col])
    
    final_output_main = pd.DataFrame() 

    for current_entity in all_legal_entities:
        entity_filtered_array = val_date_filtered_array[
            val_date_filtered_array[:, legal_entity_col] == current_entity
        ]
        if len(entity_filtered_array) == 0:
            continue
        
        product_variants = np.unique(entity_filtered_array[:, product_variant_col])

        for pv in product_variants:
            start_time2 = time.time()
            logging.warning(f"[{i}/{len(product_variants)}] Now processing PV: {pv} (start time: {start_time2})")
            try :
                final_output_main = process_product_variant(
                    product_variant_name=pv,
                    legal_entity_name=current_entity,
                    val_date_filtered_array=entity_filtered_array,
                    valuation_date=valuation_date,
                    run_id=run_id,
                    config_dict=config_dict,
                    column_index_dict=column_index_dict,
                    vol_repo_data=vol_repo_data,
                    vol_components_data=vol_components_data,
                    holiday_calendar=holiday_calendar,
                    currency_data=currency_data,
                    NMD_adjustments=NMD_adjustments,
                    repayment_schedule=repayment_schedule,
                    func=Valuation_Models.Value_extraction_pf, 
                    vix_data=vix_data,
                    cf_analysis_id=cf_analysis_id,
                    cashflow_uploaded_data=cashflow_uploaded_data,
                    underlying_position_data=underlying_position_data,
                    custom_daycount_conventions=custom_daycount_conventions,
                    dpd_ruleset=dpd_ruleset,
                    overdue_bucketing_data=overdue_bucketing_data,
                    dpd_schedule=dpd_schedule,
                    product_holiday_code=product_holiday_code,
                    request=request,
                    mtm_data=mtm_data,
                    curve_data=curve_data,
                    credit_spread_data=credit_spread_data,
                    chunk_size=chunk_size,
                    DISKSTORE_PATH=DISKSTORE_PATH,
                    final_output_main=final_output_main,
                    completion_percent=completion_percent
                )
            except Exception as e:
                logging.warning(f"Exception occurred while processing PV {pv}: {e}", exc_info=True)

            # Record the end time
            end_time2 = time.time()

            # Log the completion and time taken
            logging.warning(
                f"[{i}/{len(product_variants)}] Finished processing PV: {pv} (end time: {end_time}). "
                f"Time taken: {round(end_time2 - start_time2, 4)} seconds."
            )

    cashflow_uploaded_data = []
    del cashflow_uploaded_data

    holiday_calendar = []
    del holiday_calendar
    
    currency_data = []
    del currency_data

    NMD_adjustments = []
    del NMD_adjustments

    repayment_schedule = []
    del repayment_schedule

    mtm_data = []
    del mtm_data

    underlying_position_data = []
    del underlying_position_data

    output_dict = {}
   
    created_date = datetime.now()
    modified_date = datetime.now()
    cashflow_output_df = pd.DataFrame()
    measures_output_df = pd.DataFrame()


    logging.warning(f"   ")
    logging.warning(f"   ")
    logging.warning(f" at last  {len(val_date_filtered_array)} ")
    logging.warning(f" at last len_of_error_position_data {len(error_pos_data)} ")

    end_time = time.time()
    logging.warning(f"  end_time  {end_time}")
    diff = end_time - start_time
    logging.warning(f" Total Time Taken to run portfolio is {diff}")

    cashflow_table = config_dict['outputs']['cashflows']['save']['source']
    measure_table = config_dict['outputs']['measures']['save']['source'] 
    logging.warning(f" writing cashflow to table   {cashflow_table}")
    logging.warning(f" writing measures to table   {measure_table}")
    

    if config_dict['outputs']['cashflows']['save']['source'] != "":
        data_dir = Path(f'{DISKSTORE_PATH}Cashflow_Engine_Outputs/Cashflow')
        i = 0 
        for parquet_file in data_dir.glob(f'*_{run_id}.parquet'):
            output_df = pd.read_parquet(parquet_file)
            output_df["created_by"] = request_user
            output_df["modified_by"] = request_user
            output_df["created_date"] = created_date
            output_df["modified_date"] = modified_date
            cashflow_output_df = output_df
            # data_handling(request, output_df, config_dict['outputs']['cashflows']['save']['table'], fast_executemany=True)
            logging.warning(f" writing cashflow  {i}")
            i+=1
            output_df = []
            del output_df

        config_dict['outputs']['cashflows']['save']['source'] = ""
        config_dict['outputs']['cashflows']['save']['table'] = ""

    if config_dict['outputs']['measures']['save']['source'] != "":
        data_dir = Path(f'{DISKSTORE_PATH}Cashflow_Engine_Outputs/Measures')
        i = 0 
        for parquet_file in data_dir.glob(f'*_{run_id}.parquet'):
            output_df = pd.read_parquet(parquet_file)
            output_df["created_by"] = request_user
            output_df["modified_by"] = request_user
            output_df["created_date"] = created_date
            output_df["modified_date"] = modified_date
            measures_output_df = output_df
            # data_handling(request, output_df, config_dict['outputs']['measures']['save']['table'], fast_executemany=True)
            logging.warning(f" writing measures  {i}")
            i+=1
            output_df = []
            del output_df

        config_dict['outputs']['measures']['save']['source'] = ""
        config_dict['outputs']['measures']['save']['table'] = ""
      
    output_dict["Cashflow_Output"] = cashflow_output_df
    output_dict["Measures_Output"] = measures_output_df
    var_plot = ""

    if len(final_output_main) >  0 :
        final_output_main["Fair_Value_Per_Unit"] = final_output_main["Fair_Value_Per_Unit"].fillna("-").astype(str)
        final_output_main["Total_Holding"] = final_output_main["Total_Holding"].fillna("-").astype(str)
        final_output_main.fillna("None", inplace=True)
        final_output_main = final_output_main.head(100)

    return final_output_main, output_dict, var_plot
    

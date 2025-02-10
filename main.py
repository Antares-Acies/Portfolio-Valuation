#trigger platform reponsible for tiggering portfolio valaution's 

import logging
import pandas as pd
import numpy as np 
import os
from Portfolio_Valuation import final_valuation_fn
from config import DISKSTORE_PATH
from generate_request import RequestData
def main():

    if not os.path.exists(DISKSTORE_PATH):
        os.makedirs("output")
        os.makedirs("output/Cashflow_Engine_Outputs")
        os.makedirs("output/Cashflow_Engine_Outputs/Cashflow")
        os.makedirs("output/Cashflow_Engine_Outputs/Measures")
        os.makedirs("output/Cashflow_Engine_Outputs/Information")

    data_directory = "Portfolio Valuation Data"
    empty_data_directory = "Portfolio Valuation Sample Data"
    data_path_dict = {
        "positions_table" : "position_data.csv",
        "nmd_data" : "nmd_adjustments.csv",
        "product_data" : "product_master.csv",
        "dpd_data" : "dpd_rule_set.csv",
        "overdue_data" : "overdue_bucketing_master.csv",
        "dpd_schedule" : "dpd_schedule.csv",
        "cashflow_data_uploaded" : "cashflow_data_uploaded.csv",
        "market_data" : "quoted_security_data.csv",
        "repayment_data" : "repayment_schedule.csv",
        "product_model_mapper_table" : "product_to_model_mapping.csv",
    }

    data = {}
    for table, path in data_path_dict.items():
        try:
            full_path = os.path.join(data_directory, path)
            data[table] = pd.read_csv(full_path)
        except FileNotFoundError:
            full_path = os.path.join(empty_data_directory, path)
            data[table] = pd.read_csv(full_path)

    val_date = '2024-07-31'
    global_Var = 'Valuation Date'

    # only Yes and No
    Generate_Cashflows = 'Yes'
    Generate_Valuation = 'No'
    Generate_Sensitivity_Analysis = 'No'
    Generate_Risk_Measures = 'No'

    config_dict = {
        'function': 'Portfolio Valuation',
        'outputs': {
            'cashflows': {
                'name': 'Cashflow Data',
                'save': {
                    'source': 'existing_table',
                    'table': 'Cashflow_Data',
                    'async_mode_interim': False
                }
            },
            'measures': {
                'name': 'Measures Data',
                'save': {
                    'source': 'existing_table',
                    'table': 'Measures'
                }
            }
        },
        'inputs': {
            'data': {
                'Data1': 'importData09705342856733883',
                'Data2': 'importData05764211542493991',
                'Data3': 'importData012055534622725705',
                'Data4': 'importData06806927209892608',
                'Data5': 'importData05795491712617225',
                'Data6': 'importData0614479893854587',
                'Data7': 'renameColumn07082687671883692',
                'Data8': 'renameColumn00038268561343206553',
                'Data9': 'renameColumn07874962826394103',
                'Data10': 'importData07050988991251208'
            },
            'Output_choice': {
                'Cashflows': Generate_Cashflows,
                'Valuation': Generate_Valuation,
                'Sensitivity_Analysis': Generate_Sensitivity_Analysis,
                'Risk_Measures': Generate_Risk_Measures
            },
            'Mapper_Choice': True,
            'Data_mapper': {
                'positions': 'importData09705342856733883',
                'product_model_mapper': 'renameColumn07874962826394103',
                'product_data': 'renameColumn00038268561343206553',
                'market_data': 'importData07050988991251208',
                'nmd_data': 'importData05764211542493991',
                'cashflow_data_uploaded': 'importData012055534622725705',
                'prepayment_data': '',
                'dpd_data': 'importData0614479893854587',
                'repayment_data': 'importData06806927209892608',
                'overdue_data': 'renameColumn07082687671883692',
                'dpd_schedule': 'importData05795491712617225'
            },
            'Technical_Conf': 'Joblib',
            'Valuation_Date': {
                'val_date': val_date,
                'global_Var': global_Var
            },
            'Valuation_Method': 'risk_based_valuation',
            'Product_Model_Mapping': {},
            'VAR Methodology': '',
            'Diversified': '',
            'Lookback_Period': '',
            'Holding_Period': '',
            'Percentile': '',
            'CF_Analysis_Id': {
                'cf_analysis_id': '20th Jan 2025',
                'global_Var': ''
            },
            'col_mapping': []
        }
    }
    
    user = "anil"  # Replace with the actual user object

    request_data = RequestData('/users/App3/Build/computationModule/', '/users/App3/Build/computationModule/', '127.0.0.1:8080', user, 'http', 'Asia/Calcutta', True)


    final_valuation_fn(config_dict, request_data, data)
    return 0


if __name__ == "__main__":
    main()




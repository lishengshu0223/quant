import numpy as np
import pandas as pd
import pickle as pkl
import FactorLibrary as fl
import datetime
import warnings

warnings.filterwarnings("ignore")

df_stock_industry = pd.read_excel('.\Datalibrary\Industry\Classification.xlsx')
list_all_stock = df_stock_industry['Code'].unique().tolist()
location = './Datalibrary/Factor/'
dict_factor = {
    "consensus_profit_expectation": fl.consensus_profit_expectation,
    "enterprice_size": fl.enterprice_size,
    "profit_change": fl.profit_growth
}


def download_factors(start_date='20160101'):
    end_date = datetime.date.today().strftime("%Y%m%d")
    for factor_name, factor_fun in dict_factor.items():
        df_factor = factor_fun(list_all_stock, start_date, end_date)
        with open(fr"{location}{factor_name}.pkl", "wb") as file:
            pkl.dump(df_factor, file)
        print(f"download finish {factor_name}")
    return

if __name__ == '__main__':
    # download_factors()
    location = './Datalibrary/Factor/'
    factor_name = "profit_change"
    with open(f"{location}{factor_name}.pkl", "rb") as file:
        df_factor = pkl.load(file)
    print(df_factor)

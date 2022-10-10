import numpy as np
import pandas as pd
import pickle as pkl
import pymssql
import talib as ta
import warnings

warnings.filterwarnings("ignore")


def industry_neutralization(factor):
    """
    行业中性化、填补空缺值（1st.用股票前后两日值差值；2nd.用股票对应行业均值）
    :param factor: 因子值
    :return: 处理后的因子值
    """

    def get_param(*args, **kwargs):
        df_stock_industry = pd.read_excel('.\Datalibrary\Industry\Classification.xlsx')
        df_stock_industry.dropna(inplace=True)
        dict_stock_industry = {}
        for row in df_stock_industry.itertuples():
            dict_stock_industry[row[1]] = row[3]
        df_factor = factor(*args, **kwargs)
        df_factor.columns = ['Date', 'Code', 'Value']

        def fun1(x):
            x['Industry'] = dict_stock_industry[x.iloc[0, 1]]
            x['Value'].interpolate(inplace=True)
            x.fillna(method='ffill', inplace=True)
            return x

        df_factor = df_factor.groupby('Code').apply(lambda x: fun1(x))
        df_industry = df_factor.groupby(['Date', 'Industry'])['Value'].mean().to_frame()
        df_industry.reset_index(inplace=True)

        def fun2(x):
            x['Value'].interpolate(inplace=True)
            x.fillna(method='ffill', inplace=True)
            return x

        df_industry = df_industry.groupby(['Industry']).apply(lambda x: fun2(x))
        df_industry.columns = ['Date', 'Industry', 'Avg']
        df_factor = pd.merge(df_industry, df_factor, on=['Date', 'Industry'], how='outer')
        df_factor.loc[df_factor[df_factor['Value'].isna()].index, 'Value'] = df_factor.loc[
            df_factor[df_factor['Value'].isna()].index, 'Avg'].values
        df_factor['Factor'] = df_factor['Value'] - df_factor['Avg']
        df_factor = df_factor[['Date', 'Code', 'Factor']]
        df_factor.sort_values(['Date', 'Code'], inplace=True)
        df_factor.reset_index(inplace=True, drop=True)
        return df_factor

    return get_param


@industry_neutralization
def consensus_profit_expectation(stock_name: list, start_date='20220101', end_date='20220901'):
    """
    一致盈利预期因子
    :param stock_name: 用于测试的股票列表
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 因子值
    """
    if len(stock_name) == 1:
        stock_name = str(stock_name)
    else:
        stock_name = "'" + "', '".join(stock_name) + "'"
    conn = pymssql.connect('10.8.99.120:1510', 'winddb', 'etf@WIND#0628', 'wind_filesync', charset='cp936')
    cursor = conn.cursor()
    field = 'S_WEST_EPS_FTM_CHG_1M'
    sql = "select TRADE_DT, S_INFO_WINDCODE, convert(float, %s) from ConsensusExpectationFactor where TRADE_DT between %s and %s and S_INFO_WINDCODE in (%s)" \
          % (field, start_date, end_date, stock_name)

    cursor.execute(sql)
    df_factor = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Code', 'Factor'])
    df_factor['Date'] = pd.to_datetime(df_factor['Date'])
    df_factor['Factor'] = np.log(df_factor['Factor'])
    df_factor.sort_values(by=['Date', 'Code'], inplace=True)
    df_factor.reset_index(inplace=True, drop=True)
    return df_factor


@industry_neutralization
def enterprice_size(stock_name: list, start_date='20220101', end_date='20220901'):
    """
    市值因子
    :param stock_name:
    :param start_date:
    :param end_date:
    :return:
    """
    if len(stock_name) == 1:
        str_stock = str(stock_name)
    else:
        str_stock = "'" + "', '".join(stock_name) + "'"
    conn = pymssql.connect('10.8.99.120:1510', 'winddb', 'etf@WIND#0628', 'wind_filesync', charset='cp936')
    cursor = conn.cursor()
    sql = "select TRADE_DT, S_INFO_WINDCODE, convert(float, S_DQ_MV) " \
          "from AShareEODDerivativeIndicator where TRADE_DT between %s and %s and S_INFO_WINDCODE in (%s)" \
          % (start_date, end_date, str_stock)
    cursor.execute(sql)
    df_factor = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Code', 'Factor'])
    df_factor['Date'] = pd.to_datetime(df_factor['Date'])
    df_factor['Factor'] = np.log(df_factor['Factor'])
    df_factor.sort_values(by=['Date', 'Code'], inplace=True)
    df_factor.reset_index(inplace=True, drop=True)
    return df_factor


@industry_neutralization
def profit_growth(stock_name: list, start_date='20220101', end_date='20220901'):
    """
    营业利润增长率
    :param stock_name:
    :param start_date:
    :param end_date:
    :return:
    """
    if len(stock_name) == 1:
        str_stock = str(stock_name)
    else:
        str_stock = "'" + "', '".join(stock_name) + "'"
    conn = pymssql.connect('10.8.99.120:1510', 'winddb', 'etf@WIND#0628', 'wind_filesync', charset='cp936')
    cursor = conn.cursor()
    sql = "select TRADE_DT, S_INFO_WINDCODE, convert(float, S_DFA_TTMGROWRATE_OP) " \
          "from PITFinancialFactor where TRADE_DT between %s and %s and S_INFO_WINDCODE in (%s)" \
          % (start_date, end_date, str_stock)
    cursor.execute(sql)
    df_factor = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Code', 'Factor'])
    df_factor['Date'] = pd.to_datetime(df_factor['Date'])
    df_factor['Factor'] = np.log(df_factor['Factor'])
    df_factor.sort_values(by=['Date', 'Code'], inplace=True)
    df_factor.reset_index(inplace=True, drop=True)
    return df_factor


if __name__ == '__main__':
    df_stock_industry = pd.read_excel('.\Datalibrary\Industry\Classification.xlsx')
    df_stock_industry.dropna(inplace=True)
    dict_stock_industry = {}
    dict_industry_stock = {}
    for industry in df_stock_industry['Industry'].unique().tolist():
        dict_industry_stock[industry] = []

    for col, row in df_stock_industry.iterrows():
        list_stock = dict_industry_stock[row[2]]
        list_stock.append(row[0])
        dict_industry_stock[row[2]] = list_stock

    list_industry = ['银行', '房地产', '计算机', '环保', '商贸零售', '机械设备', '电力设备', '建筑装饰', '建筑材料', '家用电器',
                     '纺织服饰', '农林牧渔', '电子', '汽车', '公用事业', '医药生物', '综合', '石油石化', '有色金属', '通信',
                     '交通运输', '传媒', '非银金融', '基础化工', '社会服务', '轻工制造', '国防军工', '美容护理', '煤炭',
                     '食品饮料', '钢铁']
    list_all_stock = df_stock_industry['Code'].unique().tolist()
    start_date, end_date = ['20200619', '20200824']
    df_factor = profit_growth(list_all_stock, start_date, end_date)
    print(df_factor)

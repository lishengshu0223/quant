import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pkl
import pymssql
import talib as ta
import warnings
from sklearn.linear_model import LinearRegression
import FactorLibrary as fl

df_stock_industry = pd.read_excel('../行业成分/全A股票对应申万行业.xlsx')
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

start_date, end_date = ['20220601', '20220923']
list_stock = dict_industry_stock[list_industry[0]]
df_factor = fl.enterprice_size(list_stock, start_date, end_date)
df_factor = df_factor.to_frame()
df_factor.reset_index(inplace=True)
stock_name = "'" + "', '".join(list_stock) + "'"
conn = pymssql.connect('10.8.99.120:1510', 'winddb', 'etf@WIND#0628', 'wind_filesync', charset='cp936')
cursor = conn.cursor()
sql = "select TRADE_DT, S_INFO_WINDCODE, convert(float, S_DQ_PCTCHANGE) " \
      "from AShareEODPrices where TRADE_DT between %s and %s and S_INFO_WINDCODE in (%s)" \
      % (start_date, end_date, stock_name)
cursor.execute(sql)
df_data = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Asset', 'Daily_rtn'])
df_data['Date'] = pd.to_datetime(df_data['Date'])
df_data.sort_values(by='Date', inplace=True)

df_data = pd.merge(df_data, df_factor, on=['Date', 'Asset'])

model = LinearRegression()
list_coef = []
for date in df_data['Date'].unique():
    df_temp = df_data[df_data['Date'] == date][['Daily_rtn', 'Factor']]
    y_train = df_temp[['Daily_rtn']].values
    x_train = df_temp[['Factor']].values
    model.fit(x_train, y_train)
    list_coef.append(model.coef_[0][0])

df_factor_rtn = pd.DataFrame(data=np.array([list_coef]).T, columns=['Coef'], index=df_data['Date'].unique())
df_factor_rtn.reset_index(inplace=True)
df_stock = df_data[df_data['Asset'] == '000001.SZ'][['Date', 'Daily_rtn']]
df_stock = pd.merge(df_stock, df_factor_rtn, left_on=['Date'], right_on=['index'])
print(df_stock)


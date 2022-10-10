import dateutil.relativedelta
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import datetime
import pymssql
import warnings
import FactorLibrary
import alphalens as al

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def get_stock_data(stock_name: list, start_date='20220101', end_date='20220901'):
    """
    获取股票数据
    :param stock_name:股票列表
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
    sql = "select TRADE_DT, S_INFO_WINDCODE, convert(float, S_DQ_PCTCHANGE) from AShareEODPrices where TRADE_DT between %s and %s and S_INFO_WINDCODE in (%s)" \
          % (start_date, end_date, str_stock)

    cursor.execute(sql)
    df_stock = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Code', 'Daily_rtn'])
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    df_stock.sort_values(['Date', 'Code'], inplace=True)
    return df_stock


def stock_select(df: pd.DataFrame,
                 list_stock: list,
                 stock_pool=None,
                 nonewlyIPO=False,
                 noST=False,
                 start_date='20210101',
                 end_date='20211231'):
    """

    :param df: 已经选择的股票
    :param list_stock: 预计筛选的股票列表
    :param stock_pool: 所筛选股票池（中证500，沪深300）
    :param nonewlyIPO: 是否剔除IPO
    :param noST: 是否剔除ST
    :param start_date:
    :param end_date:
    :return:
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    days = 0
    conn = pymssql.connect('10.8.99.120:1510', 'winddb', 'etf@WIND#0628', 'wind_filesync', charset='cp936')
    cursor = conn.cursor()
    if stock_pool is None:
        df_constituent_stock = pd.DataFrame(columns=['Date_in', 'Date_out', 'Index', 'Code'])
    else:
        sql = "select S_CON_INDATE,S_CON_OUTDATE, S_INFO_WINDCODE, S_CON_WINDCODE from AIndexMembers " \
              "where S_INFO_WINDCODE in ('%s')" % stock_pool
        cursor.execute(sql)
        df_constituent_stock = pd.DataFrame(data=cursor.fetchall(), columns=['Date_in', 'Date_out', 'Index', 'Code'])
        df_constituent_stock['Date_in'] = pd.to_datetime(df_constituent_stock['Date_in'])
        df_constituent_stock['Date_out'] = pd.to_datetime(df_constituent_stock['Date_out'])

    if nonewlyIPO is False:
        df_ipo = pd.DataFrame(columns=['Date', 'Code'])
    else:
        sql = "select S_INFO_LISTDATE, S_INFO_WINDCODE from AShareDescription"
        cursor.execute(sql)
        df_ipo = pd.DataFrame(data=cursor.fetchall(), columns=['Date', 'Code'])
        df_ipo['Date'] = pd.to_datetime(df_ipo['Date'])
        df_ipo.dropna(inplace=True)

    if noST is False:
        df_st = pd.DataFrame(columns=['Date_in', 'Date_out', 'Code', 'TYPE'])
    else:
        sql = "select ENTRY_DT, REMOVE_DT, S_INFO_WINDCODE, S_TYPE_ST from AShareST"
        cursor.execute(sql)
        df_st = pd.DataFrame(data=cursor.fetchall(), columns=['Date_in', 'Date_out', 'Code', 'TYPE'])
        df_st['Date_in'] = pd.to_datetime(df_st['Date_in'])
        df_st['Date_out'] = pd.to_datetime(df_st['Date_out'])
    dict_date_stock = {}
    for date in date_range:
        if stock_pool is not None:
            list_pool = df_constituent_stock[(df_constituent_stock['Date_in'] <= pd.to_datetime(date)) &
                                             ((df_constituent_stock['Date_out'] >= pd.to_datetime(date)) |
                                              (df_constituent_stock['Date_out'].isna()))]['Code'].to_list()
        else:
            list_pool = list_stock
        list_ipo = df_ipo[(df_ipo['Date'] >= (date - pd.Timedelta(days=days))) &
                          (df_ipo['Date'] <= date)]['Code'].to_list()
        list_st = df_st[(df_st['Date_in'] <= date) &
                        ((df_st['Date_out'] >= date) | (df_st['Date_out'].isna())) &
                        (df_st['TYPE'].str.contains('S|T|L'))]['Code'].to_list()
        list_remained = list(set(list_stock) & (set(list_pool) - (set(list_ipo) | set(list_st))))
        dict_date_stock[date] = list_remained
    list_date = []
    list_asset = []
    list_factor = []
    df.columns = ['Date', 'Asset', 'Value']
    for date in df['Date'].unique():
        df_temp = df[df['Date'] == date]
        df_temp = df_temp[df_temp['Asset'].isin(dict_date_stock[pd.to_datetime(date)])]
        list_date.append(df_temp['Date'].to_list())
        list_asset.append(df_temp['Asset'].to_list())
        list_factor.append(df_temp['Value'].to_list())

    list_date = [item for sublist in list_date for item in sublist]
    list_asset = [item for sublist in list_asset for item in sublist]
    list_factor = [item for sublist in list_factor for item in sublist]
    df = pd.DataFrame(data=np.array([list_date, list_asset, list_factor]).T, columns=['Date', 'Code', 'Value'])
    return df


class Test:
    def __init__(self, list_stock, start_date, end_date, stock_pool=None, nonewlyIPO=False, noST=False):
        self.list_stock = list_stock
        self.start_date = start_date
        self.end_date = end_date
        self.stock_pool = stock_pool
        self.nonewIPO = nonewlyIPO
        self.noST = noST
        self.factor = None
        self.ts_ic = None
        self.quantile_dailyrtn = None
        self.quantile_cumrtn = None

        self.stock = get_stock_data(stock_name=self.list_stock,
                                    start_date=self.start_date,
                                    end_date=self.end_date)
        self.stock = stock_select(df=self.stock,
                                  list_stock=self.list_stock,
                                  stock_pool=self.stock_pool,
                                  nonewlyIPO=self.nonewIPO,
                                  noST=self.noST,
                                  start_date=self.start_date,
                                  end_date=self.end_date)

    def ic_test(self, df_factor, rank=True):
        df_factor = stock_select(df=df_factor,
                                 list_stock=self.list_stock,
                                 stock_pool=self.stock_pool,
                                 nonewlyIPO=self.nonewIPO,
                                 noST=self.noST,
                                 start_date=self.start_date,
                                 end_date=self.end_date)
        if rank:
            df_factor.set_index(['Date', 'Code'], inplace=True)
            df_factor.groupby('Date').rank()
            df_factor.reset_index(inplace=True)
        self.factor = df_factor.copy(deep=True)
        df_factor_stock = pd.merge(self.factor, self.stock, on=['Date', 'Code'], how='inner')
        df_factor_stock = df_factor_stock[['Date', 'Value_x', 'Value_y']]
        df_factor_stock.fillna(method='bfill', inplace=True)
        df_ic = df_factor_stock.groupby('Date').corr().unstack()
        df_ic = df_ic.iloc[:, 1]
        df_ic = df_ic.to_frame()
        df_ic.columns = ['IC_value']
        self.ts_ic = df_ic.copy(deep=True)
        del df_factor, df_factor_stock, df_ic
        return self.ts_ic

    def plot_ts_ic(self, df_factor=None):
        if self.ts_ic is None or df_factor is not None:
            self.ic_test(df_factor)
        y_ticks = np.arange(-1, 1, 0.1)
        y_ticks = y_ticks[abs(y_ticks) < abs(self.ts_ic).max().values[0]]
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(self.ts_ic, linewidth=0.5, alpha=0.5)
        plt.plot(self.ts_ic.rolling(22).mean(), linewidth=1)
        plt.axhline(y=0, linewidth=1.5, color='black')
        plt.yticks(y_ticks)
        plt.xlim(self.ts_ic.index.min(), self.ts_ic.index.max())
        plt.text(x=0.01, y=0.87, s="  均值:%.4f\n标准差:%.4f" % (self.ts_ic.mean().values[0], self.ts_ic.std().values[0]),
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        plt.legend(['日波动', '月均线'], loc='upper right')
        plt.show()

    def monotonicity_test(self, df_factor, quantile=5):
        df_factor = stock_select(df=df_factor,
                                 list_stock=self.list_stock,
                                 stock_pool=self.stock_pool,
                                 nonewlyIPO=self.nonewIPO,
                                 noST=self.noST,
                                 start_date=self.start_date,
                                 end_date=self.end_date)
        df_quantile = df_factor.copy(deep=True)
        df_quantile.set_index(['Date', 'Code'], inplace=True)

        def fun(x):
            for i in range(1, quantile + 1):
                x['temp'] = x['Value'].quantile(i / quantile)
                x[i] = 0
                x[i][x['Value'] > x['temp']] = 1
            return x

        df_quantile = df_quantile.groupby("Date").apply(lambda x: fun(x))
        df_quantile['Quantile'] = df_quantile[[i for i in range(1, quantile + 1)]].sum(axis=1) + 1
        df_quantile.reset_index(inplace=True)
        df_quantile = df_quantile[['Date', 'Code', 'Value', 'Quantile']]
        df_quantile = pd.merge(df_quantile, self.stock, on=['Date', 'Code'], how='inner')
        df_quantile.set_index(['Date', 'Code'], inplace=True)
        df_quantile = df_quantile[['Quantile', 'Value_y']]
        df_quantile = df_quantile.groupby(['Date', 'Quantile'])['Value_y'].mean()
        df_quantile = df_quantile.to_frame()
        df_quantile.reset_index(inplace=True)
        df_quantile = df_quantile.pivot(columns=['Quantile'], index=['Date'])
        df_quantile.columns = [1, 2, 3, 4, 5]
        self.quantile_dailyrtn = df_quantile.copy(deep=True)
        df_quantile = df_quantile / 100 + 1
        df_quantile = df_quantile.cumprod()
        df_quantile = df_quantile / df_quantile.iloc[0, :].values
        self.quantile_cumrtn = df_quantile.copy(deep=True)
        return self.quantile_cumrtn

    def plot_monotonicity(self, df_factor, quantile=5):
        if self.quantile_cumrtn is None or df_factor is not None:
            self.monotonicity_test(df_factor, quantile)
        plt.figure(figsize=(10, 4), dpi=100)
        plt.plot(self.quantile_cumrtn, linewidth=0.5)
        plt.legend(self.quantile_cumrtn.columns)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    df_stock_industry = pd.read_excel('.\Datalibrary\Industry\Classification.xlsx')
    df_stock_industry.dropna(inplace=True)
    list_test = df_stock_industry['Code'].to_list()
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
    start_date = '20130101'
    end_date = '20150901'
    num = 500
    df_factor = FactorLibrary.enterprice_size(stock_name=list_test, start_date=start_date, end_date=end_date)
    test = Test(list_stock=list_test, stock_pool=None, nonewlyIPO=True, noST=True, start_date=start_date,
                end_date=end_date)
    df_ic = test.ic_test(df_factor)
    # print(df_ic)
    test.plot_ts_ic()
    # test.plot_monotonicity(df_factor)

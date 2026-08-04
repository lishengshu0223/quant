import os

_DATA_ROOT = r"F:\Trade_data"

STOCK_PRICE_DIR = "stock_price"
INDEX_PRICE_DIR = "index_price"
INDEX_WEIGHTS_DIR = "index_weights"
BARRA_DIR = "barra"
INSTRUMENTS_DIR = "instruments"
TRADING_DATES_DIR = "trading_dates"
STOCK_MINUTE_DIR = "rq_backtest_data"
BUNDLE_H5_DIR = "bundle"
EQUITIES_H5_DIR = "h5"

STOCK_PRICE_FIELDS = ["open", "close", "high", "low", "total_turnover", "volume"]
ADJUSTED_FIELDS = ["adjopen", "adjclose", "adjhigh", "adjlow", "adjvolume"]
STOCK_MINUTE_FIELDS = ["open", "high", "low", "close", "volume", "total_turnover", "num_trades"]
MINUTE_FREQUENCIES = ["1m", "5m", "15m", "30m", "60m"]

# Bundle数据文件路径常量
BUNDLE_DIR = "rq_backtest_data"
BUNDLE_BUNDLE_DIR = "bundle"
FUTURES_H5 = "futures.h5"
FUNDS_H5 = "funds.h5"
DIVIDENDS_H5 = "dividends.h5"
EX_CUM_FACTOR_H5 = "ex_cum_factor.h5"
SPLIT_FACTOR_H5 = "split_factor.h5"
ST_STOCK_DAYS_H5 = "st_stock_days.h5"
SUSPENDED_DAYS_H5 = "suspended_days.h5"
TRADABLE_STATUS_DIR = "tradable_status"
YIELD_CURVE_H5 = "yield_curve.h5"
FUTURE_INFO_JSON = "future_info.json"
INSTRUMENTS_PK = "instruments.pk"
FUND_INSTRUMENTS_PKL = "fund_instruments.pkl"
CONVERTIBLE_INSTRUMENTS_PKL = "convertible_instruments.pkl"
OPTION_INSTRUMENTS_PKL = "option_instruments.pkl"

FUTURES_DAILY_FIELDS = ["open", "close", "high", "low", "volume", "total_turnover",
                        "settlement", "prev_settlement", "open_interest", "prev_close",
                        "limit_up", "limit_down"]
FUTURES_MINUTE_FIELDS = ["open", "high", "low", "close", "volume", "total_turnover", "open_interest"]
FUNDS_DAILY_FIELDS = ["open", "close", "high", "low", "volume", "total_turnover",
                      "prev_close", "limit_up", "limit_down"]
DIVIDEND_FIELDS = ["book_closure_date", "announcement_date", "dividend_cash_before_tax",
                   "ex_dividend_date", "payable_date", "round_lot"]
YIELD_CURVE_TENORS = ["0S", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y",
                      "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "15Y", "20Y", "30Y", "40Y", "50Y"]

INDEX_CODES = {
    "000016": "上证50",
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "932000": "中证2000",
    "866006": "微盘股指数",
}

INDEX_FULL_CODES = {
    "000016": "000016.XSHG",
    "000300": "000300.XSHG",
    "000905": "000905.XSHG",
    "000852": "000852.XSHG",
    "932000": "932000.CSI",
    "866006": "866006.RI",
}

INDEX_WEIGHT_CODES = {k: v for k, v in INDEX_FULL_CODES.items() if k != "932000"}

FACTOR_MODELS = {
    "v1": ["Liquidity", "Leverage", "BTOP", "Earnings Yield", "Growth", 
           "Momentum", "Size", "Beta", "Mid cap", "Profitability"],
    "v2": ["Liquidity", "Leverage", "BTOP", "Earnings Yield", "Growth", 
           "Momentum", "Size", "Beta", "Mid cap", "Profitability",
           "Investment Quality", "Earning Variablity", "Earnings Quality",
           "Long Term reversal", "Resival Volatility", "Dividend Yield"],
}

INDUSTRY_MAPPING = "sws_2021"


def init(data_root=None):
    global _DATA_ROOT
    if data_root is not None:
        _DATA_ROOT = data_root


def get_data_root():
    return _DATA_ROOT


def get_data_path(data_type, *subdirs):
    paths = [_DATA_ROOT, data_type]
    paths.extend(subdirs)
    return os.path.join(*paths)

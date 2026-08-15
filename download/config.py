import os

DEFAULT_DATA_ROOT = r"F:\Trade_data"

STOCK_PRICE_DIR = "stock_price"
INDEX_PRICE_DIR = "index_price"
INDEX_WEIGHTS_DIR = "index_weights"
BARRA_DIR = "barra"
INSTRUMENTS_DIR = "instruments"
TRADING_DATES_DIR = "trading_dates"
ANNOUNCEMENTS_DIR = "announcements"
TURNOVER_DIR = "turnover"
SHARES_DIR = "shares"

STOCK_PRICE_FIELDS = ["open", "close", "high", "low", "total_turnover", "volume"]

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
    "932000": "932000.INDX",
    "866006": "866006.RI",
}

INDEX_WEIGHT_CODES = dict(INDEX_FULL_CODES)

FACTOR_MODELS = {
    "v1": ["Liquidity", "Leverage", "BTOP", "Earnings Yield", "Growth", 
           "Momentum", "Size", "Beta", "Mid cap", "Profitability"],
    "v2": ["Liquidity", "Leverage", "BTOP", "Earnings Yield", "Growth", 
           "Momentum", "Size", "Beta", "Mid cap", "Profitability",
           "Investment Quality", "Earning Variablity", "Earnings Quality",
           "Long Term reversal", "Resival Volatility", "Dividend Yield"],
}

INDUSTRY_MAPPING = "sws_2021"
START_DATE = "2016-01-01"


def get_data_path(data_type, *subdirs):
    paths = [DEFAULT_DATA_ROOT, data_type]
    paths.extend(subdirs)
    return os.path.join(*paths)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import datetime
import pymssql
import warnings
import FactorLibrary
import ValidilityTest
import alphalens as al

a = [1,2,3,4,5]
b = [1,2,3,4]
c = [1,2]
d = [3]
e = list(set(a) & (set(b) - (set(c) | set(d))))
print(e)
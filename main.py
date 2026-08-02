import rqdatac 
rqdatac.init()




data = rqdatac.get_factor_exposure(["000001.XSHE"], start_date="2023-10-23", end_date="2023-10-24", model="v2")
print(data.shape)
print(data.columns)
print(data)

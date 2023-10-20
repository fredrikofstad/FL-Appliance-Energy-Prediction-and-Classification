import pandas as pd

data_path = "../data/"

xlsx = pd.ExcelFile(data_path + 'dataset.xlsx')
sheets = xlsx.sheet_names

# takes too long to parse excel sheets
# TODO: convert to csv and read that way

#consumer_data = {}
#index = 1
#for index, sheet in enumerate(sheets):
#    consumer_data[index] = xlsx.parse(sheet)

#print(consumer_data[3])

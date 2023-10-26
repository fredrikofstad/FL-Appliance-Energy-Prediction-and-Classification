import pandas as pd

data_path = "../data/"

excel_file = pd.ExcelFile(data_path + 'dataset.xlsx')

# Get the names of all sheets in the Excel file.
sheet_names = excel_file.sheet_names

for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)

    csv_file_name = f'{data_path}{sheet_name}.csv'
    df.to_csv(csv_file_name, index=False)

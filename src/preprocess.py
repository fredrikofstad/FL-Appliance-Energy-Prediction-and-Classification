import pytablereader as ptr
import pytablewriter as ptw

file_path = "sample_data.csv"

with open(file_path, "w") as f:
    f.write(csv_text)

loader = ptr.CsvTableFileLoader(file_path)

for table_data in loader.load():
    print("\n".join([
        "load from file",
        "===============",
        "{:s}".format(ptw.dumps_tabledata(table_data)),
    ]))

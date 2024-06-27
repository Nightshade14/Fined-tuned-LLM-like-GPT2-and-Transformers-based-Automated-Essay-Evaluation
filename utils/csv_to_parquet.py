import sys
import pandas as pd
filePath = sys.argv[1]
try:
	df = pd.read_csv(filePath+"train.csv")
	df.to_parquet(filePath+"train.parquet", index=False)
	print("File converted successfully")
except:
	print("error")
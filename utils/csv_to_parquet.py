import argparse
import traceback
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--src_file", help="Source file path (csv)")
parser.add_argument("-d", "--dest_file", help="Destination file path (parquet)")
args = parser.parse_args()

if not args.dest_file:
	args.dest_file = args.src_file[:-3] + "parquet"

try:
	df = pd.read_csv(args.src_file)
	df.to_parquet(args.dest_file, index=False)
	print("File converted successfully")
except:
	traceback.print_exc()
finally:
	print("Program execution completed.")
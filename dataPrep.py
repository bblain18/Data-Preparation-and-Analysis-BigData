# Notes:
#   Data set: Soil temperature of some location for the month of September (9)
#   Labels: Day
#   Features/Values: Temperature
#
#
#
#   Data preparation:
#       1. Create and label values
#       2. Iterate over rows in CSV & split "labels" and "values". Split rows based upon ',' into an array
#          This array will have a label at [0] and value at [1] with our pariticular dataset
#       3. Form matrix/matrices out of array, using numpy
#   Data cleaning:
#       - Remove all null values/replace them with 0s
#       - Remove Time, Data, Year from data/row[0] (should be able to use .Month)
#       - Ensure temperature/row[1] is a float
#       - Remove file "headers" (ex: Timestamp, Measurement interval, etc) from dataset and output to log fo analysis
#   Analysis:
#       - Avg/day
#       - Avg over month
#       - Varience/day
#       - Varience over month
#       - Std Deviation/day
#       - Std Deviation over month
#       - MADD/day
#       - MADD over month

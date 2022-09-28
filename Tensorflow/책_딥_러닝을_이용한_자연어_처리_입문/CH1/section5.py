import pandas as pd
import pandas_profiling
data =pd.read_csv('../dataset/spam.csv', encoding='latin1')
print(data.head(5))

pr = data.profile_report()
pr.to_file("./pr_report.html")
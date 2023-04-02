import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('c:\\tmp\\gmd.csv',sep=';')

profile = ProfileReport(df, title="Profiling Report")

profile.to_file("c:\\tmp\\gmd_profiling_dataset01.html")

print('Terminado')

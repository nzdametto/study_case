import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

def report():
    df = pd.read_csv("case_study_anonymized.csv", sep="|", encoding="latin")
    profile = ProfileReport(df)
    report_file = "dataset_report.html"
    profile.to_file(report_file)
    return report_file
	 

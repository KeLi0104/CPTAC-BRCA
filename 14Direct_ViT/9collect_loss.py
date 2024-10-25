
import os,sys
import numpy as np
import pandas as pd
from PyPDF2 import PdfFileMerger, PdfFileReader

ik_folds = [0,1,2,3,4]
il_folds = [0,1,2,3,4]

mergedObject = PdfFileMerger()

for ik_fold in ik_folds:
    for il_fold in il_folds:
        mergedObject.append(PdfFileReader(f"results/result_{ik_fold}_{il_fold}/loss.pdf", "rb"))
        
mergedObject.write("loss.pdf")
print("--- completed --- ")
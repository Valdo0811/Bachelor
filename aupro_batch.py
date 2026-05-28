import sys
import glob
import torch
from anomalib.metrics import AUROC, PRO, AUPRO

folder = sys.argv[1] + "/*/*" + ".pt"
    
prompts = []
preds = {}    
f = open("runs.bat", "a")    
for filename in sorted(glob.glob(folder)):
        x = filename.split('\\')
        x = x[-1].split('.')
        x = x[0]
        filename = filename.replace("\\", "/")
        f.write("python aupro.py " + filename + "\n")
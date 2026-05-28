import sys
import glob
import torch
import os
from anomalib.metrics import AUROC, PRO, AUPRO

folder = sys.argv[1] + "/*/*" + ".pt"
f = open("runs.bat", "a")
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]

for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        for sub in subs:
                f.write("python multi_figure_auroc.py " + sys.argv[1] + "/" + dir + "/" + sub + "\n")

#subsubs = [os.path.(path) for path in glob.glob(f'{sys.argv[1]}/{subs}/*') for subs in subdirectories]
#print(subdirectories) 
'''    
prompts = []
preds = {}    
f = open("runs.bat", "a")    
for filename in sorted(glob.glob(folder)):
        x = filename.split('\\')
        x = x[-1].split('.')
        x = x[0]
        filename = filename.replace("\\", "/")
        f.write("python multi_figure_auroc.py " + filename + "\n")'''
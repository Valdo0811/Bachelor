import sys
import glob
import torch
import os
from anomalib.metrics import AUROC, PRO, AUPRO

folder = sys.argv[1] + "/*/*" + ".pt"
f = open("prompt.json", "a")
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
f.write("{\n")
for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        f.write(f"\"{dir}\": " + "{\n")
        for sub in subs:
                f.write(f"\"{sub}\": [\"broken\", \"damage\", \"damaged\"],\n")
        f.write("},\n")
f.write("}\n")

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
import sys
import glob
import torch
import os
from anomalib.metrics import AUROC, PRO, AUPRO

folder = sys.argv[1] + "/*/*" + ".pt"
f = open("multi_fig_run.bat", "w")
categories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
for category in categories:
        error_types = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{category}/*')]
        for error_type in error_types:
                f.write(f"python multi_figure.py metrics/{category}/{error_type}\n")
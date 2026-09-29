import sys
import glob
import os

generic_prompts = "[\"broken\", \"damage\", \"damaged\"]"

folder = sys.argv[1] + "/*/*" + ".pt"
f = open("prompts.json", "w")
subdirectories = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/*')]
f.write("{\n")
for dir in subdirectories:
        subs = [os.path.basename(path) for path in glob.glob(f'{sys.argv[1]}/{dir}/*')]
        f.write(f"\"{dir}\": " + "{\n")
        for sub in subs:
                f.write(f"\"{sub}\": {generic_prompts},\n")
        f.write("},\n")
f.write("}\n")

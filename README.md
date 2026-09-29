# Setup

create a conda env from the environment.yml file with

'''
conda env create -f environment.yml
'''

switch to the env with

conda activate sam3

install PyTorch 

pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

clone the sam3 repository

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

download optional dependencies for faster inference

pip install einops ninja && pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/ronghanghu/cc_torch.git

Request access to the Hugging Face repo to download the checkpoints, see: https://github.com/facebookresearch/sam3#getting-started

# Prompts

Enter the prompts you want to use for each error type in the prompts.json file.

If you want to use generic prompts for every error type, you can edit them in generate_generic_prompts.py and run it. Do this before adding the specific ones!



# Run the framework

For single runs use 

python experiment.py --config config.yaml

To run every prompt use


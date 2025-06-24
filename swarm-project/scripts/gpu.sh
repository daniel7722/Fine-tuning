#!/bin/bash
# SBATCH --gres=gpu:1
# SBATCH --mail-type=ALL # required to send email notifcations
# SBATCH --mail-user=dh320 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/${USER}/tiny/bin/:$PATH
# the above path could also point to a miniconda install
# if using miniconda, uncomment the below line
# source ~/.bashrc
source activate
source /vol/cuda/12.4.0/setup.sh
/usr/bin/nvidia-smi
uptime
python3 Fine-tuning/swarm-project/python_sims/model.py
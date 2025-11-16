To open Carla GUI

1.) open terminal when in quest via FastX
2.) cd /projects/p33058/carla
3.) srun --x11 -A p33058 -p gengpu --gres=gpu:a100:1 --mem=16G -t 1:00:00 --pty bash -l
4.) conda activate /projects/p33058/envs/carla-py39
5.) ./CarlaUE4.sh





# 338-ride


## Setup (with Northwestern Quest HPC)
### If you have a Mac 
Install FastX3 at https://www.starnet.com/download/fastx-client. Log in with your Northwestern <NUID>@quest.northwestern.edu, password.
View this helpful Northwestern guide: https://services.northwestern.edu/TDClient/30/Portal/KB/ArticleDet?ID=1511#config1
Video Resource: https://www.youtube.com/watch?v=xFYHs19gGjU

The default FastX3 hardware is not sufficient to run CARLA. To acquire an A100 GPU, run:
```
groups
```
The result should be your allocation ID (e.g. eXXXXX). Then run:
```
srun -A eXXXXX -p gengpu --mem=12G --gres=gpu:a100:1 -N 1 -n 1 -t <time> --pty bash -l
```

## After getting Windows/Linux
Follow (instructions)[https://carla.readthedocs.io/en/latest/start_quickstart/#download-and-extract-a-carla-package] to install the most recent stable CARLA package.

Make sure your Python version is acceptable (FastX3 has version 3.8), and create a virtual environment with that version: 
```
python -m venv carla                   # to create a venv named "carla"
source carla/bin/activate               # to activate the venv
python -m pip install carla
python -m pip install --upgrade pip     # to upgrade pip
cd CarlaUE4/PythonAPI/examples
pip install -r requirements.txt         # install required packages
```

Then navigate back to the home directory and run 
```
./CarlaUE4.sh
```
to start the program. The screen will be black for some time before starting.

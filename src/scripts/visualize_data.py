import torch 
import wandb 
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

 
wandb_run = wandb.init(project="amass_after_process", entity="hmp",  dir="./logs/hands_after_process", name="amass data")
# writer = SummaryWriter("./logs/hands_after_process")
f = open('./logs/hands_after_process/amass_eligible_mocaps.txt', 'r').read().split(" \n")

rh_var, lh_var, all_hand_var = [], [], []

for line in f:
    try:
        nums = line.split(": ")[1].split(",") 
    except:
        nums = [0, 0]
        
    r_var = float(nums[0])
    l_var = float(nums[1])
    
    rh_var.append(r_var)
    lh_var.append(l_var)
    
    all_hand_var.append(r_var + l_var)

rh_var = [[elem] for elem in rh_var]
lh_var = [[elem] for elem in lh_var]
all_hand_var = [[elem] for elem in all_hand_var]


wandb_run.log({'right_hand': wandb.plot.histogram(wandb.Table(data=rh_var, columns=["scores"]), "scores", title="right_hand"),
        'left_hand': wandb.plot.histogram(wandb.Table(data=lh_var, columns=["scores"]), "scores", title=None)})
wandb_run.log({'all_hand': wandb.plot.histogram(wandb.Table(data=all_hand_var, columns=["scores"]), "scores", title=None)})

# wandb_run.log({"right": wandb.plot.histogram(rh_var, "var"), "left": wandb.Histogram(lh_var), "all": wandb.Histogram(np_histogram=all_hand_var)})

# writer.add_histogram("left_hand_var", lh_var)
# writer.add_histogram("right_hand_var", rh_var)

# plt.hist(rh_var, bins=50, alpha=0.5, label='right hand')
# plt.grid()
# plt.savefig("./logs/hands_after_process/rh_var_hist.png")
# plt.clf()
# plt.hist(lh_var, bins=50, alpha=0.5, label='left hand')
# plt.grid()
# plt.savefig("./logs/hands_after_process/lh_var_hist.png")
# plt.clf()
# plt.hist(all_hand_var, bins=50, alpha=0.5, label='all hand')
# plt.grid()
# plt.savefig("./logs/hands_after_process/hand_var_hist.png")

# import ipdb; ipdb.set_trace()


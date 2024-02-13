import sys
sys.path.append('./../')
from re import A
import pandas as pd
import numpy as np
import openai
import argparse
import torch
import time, json, os

# from utils.classification_data_generator import df2jsonl,array2prompts
from utils.classificationDataGen import dataGenerate
from utils.helper import log
import utils.configs as cfgs
from models.GPTJFineTuner import GPTJFineTuner
from models.baselines import clf_model_teachers
from run_exps_helper import *
from utils.train_nn_generator import DataNN
from utils.helper import add_ending_symbol
from models import lora_gptj


pid = os.getpid()
print("pid:",pid) 

parser = argparse.ArgumentParser(description='GPTJ')
parser.add_argument("-b","--batch_size", default=4, type=int)
parser.add_argument("-d","--data_id", default=-1, type=int)
parser.add_argument("-f","--use_feature_name", action='store_true',help='replace feature indices with feature names')
parser.add_argument("-g","--gpu_id", default=0, type=int)
parser.add_argument("-l","--lr", default=0.1, type=float,help="learning rate multiplier of GPT3, or learning rate of GPTJ")
parser.add_argument("-p", "--epochs", default=10, type=int)
parser.add_argument("-s", "--seed", default=1, type=int)


args = parser.parse_args()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(args.seed)

args = parser.parse_args()

CUDA_ID = args.gpu_id
LABEL_NOISES = [0.05, 0.1, 0.2]


######## Data
done_keys = []
data_ids = [args.data_id] if args.data_id > -1 else cfgs.openml_data_ids.keys()
lst_ids = []
for k in data_ids:
    if k not in done_keys:
        lst_ids.append(k)
data_ids = np.sort(lst_ids)

# set up saving path
log_fpath = f".../results/evals/run_gptj_featname_{args.use_feature_name}.txt"
logf = open(log_fpath, 'a+')

CONFIG = {'learning_rate': args.lr, 'batch_size': args.batch_size, 'epochs':[args.epochs], 'weight_decay': 0.01, 'warmup_steps': 6}


for data_id in data_ids:
    
    csv_fpath = f"results/csv/run_gptj_featname_{args.use_feature_name}_did_{args.data_id}_runidx.csv"
    # load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id))
    train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
    train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
    jsonl_files = load_jsonl(data_id,args.use_feature_name) # load w/ or w/o feature names

    
    val_prompts = extract_prompts(jsonl_files['val'])
    test_prompts = extract_prompts(jsonl_files['test'])
    
    print("Training and Validation jsonl:", jsonl_files)


    #Select 'new_model' new_model = False or new_model = True
    gptj_fine_tuner = GPTJFineTuner(config=CONFIG, train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
    gptj_fine_tuner.fine_tune()
    log(logf, f"==== DID {data_id} run===")            
    gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
            valid_prompts = val_prompts, 
            valid_df = val_df,
            test_prompts = test_prompts, 
            test_df = test_df, 
            train_df = train_df,
            logf=logf
        )
    try:
        pd.DataFrame(gptj_best_test_y).to_csv(csv_fpath)
    except:
        print("!!!! can't save gptj_best_test_y")

import sys
sys.path.append('./')
sys.path.append('./../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import get_linear_schedule_with_warmup, AdamW

from bitsandbytes.optim import Adam8bit

import os, time
from tqdm import tqdm
import json
import numpy as np
import gc 
gc.collect()

from models.lora_gptj_ops import GPTJForCausalLM, GPTJBlock, add_adapters

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GPTJDataset(Dataset):
    def __init__(self, json_lst, tokenizer, max_length=1024):
        texts = []
        completion_lens = []
        for row in json_lst:
            t = ' '.join(row.values())
            texts.append(t)
            l = len(tokenizer.tokenize(row['completion']))
            completion_lens.append(l)
        # print(completion_lens)
        tokens = tokenizer(texts, truncation=True, padding = True, max_length=max_length, return_tensors='pt')
        self.input_ids = tokens['input_ids']
        self.attention_mask = tokens['attention_mask']
        self.labels = []
        for i in range(len(self.input_ids)):
            b_labels = self.input_ids[i].clone()
            b_labels[:-completion_lens[i]] = -100
            self.labels.append(b_labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx] 

class LoRaQGPTJ:
    def __init__(self, model_name='EleutherAI/gpt-j-6B', adapter=True, device=None, model_path='../results/gpt-j/') -> None:
        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J
        self.config = transformers.GPTJConfig.from_pretrained("EleutherAI/gpt-j-6B")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
            
        # Define PAD Token = EOS Token = 50256 -- new modifications
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.use_cache = False
        
        # finetune
        if adapter:
            add_adapters(self.model)

        # if not(model_name == 'EleutherAI/gpt-j-6B'):
        #     self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        #     self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)

        self.device = device
        self.model = self.model.to(self.device)
        self.model_path = model_path
    
    # def load_networks(self, model_name):
    #     self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).to(self.device)

    def prepare_data(self, jsonl_path):
        with open(jsonl_path, 'r') as json_file:
            json_lst = list(json_file)

        txt_list = []
        for json_str in json_lst:
            result = json.loads(json_str)
            txt_list.append(result)
        
        data = GPTJDataset(txt_list, self.tokenizer)
        return data

    def finetune(self, train_jsonl_path, val_jsonl_path, train_configs={'batch_size': 8, 'epochs': 20, 'learning_rate': 1e-3, 'weight_decay': 0.01, 'warmup_steps': 20}, saving_checkpoint=True):
        train_data = self.prepare_data(train_jsonl_path)
        val_data = self.prepare_data(val_jsonl_path)
        data_loader = DataLoader(train_data, batch_size=train_configs['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_configs['batch_size'])
        total_steps = len(data_loader) * train_configs['epochs']
        # params 
        params_for_optimizer = []
        for name, param in self.model.named_parameters():
            if "adapter" in name: # "attn" in name and 
                param.requires_grad = True
                params_for_optimizer.append(param)

                # nn.init.zeros_(param)
                # nn.init.xavier_uniform_(param)
            else:
#                 print(f"Setting {name} requires_grad=False")
                param.requires_grad = False
        self.model.gradient_checkpointing_enable()

        optimizer = Adam8bit(params_for_optimizer, lr=train_configs['learning_rate'], weight_decay=0.01) # freeze the W
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = train_configs['warmup_steps'], 
                                            num_training_steps = total_steps)

        best_loss = np.inf
        self.train_loss_list, self.val_loss_list = [],[]
        # with torch.cuda.amp.autocast():
        for epoch in range(train_configs['epochs']):
            # self.model.train()
            tqdm_object = tqdm(data_loader, total=len(data_loader), desc=f"Epoch: {epoch + 1}")
            loss_meter = AverageMeter()
            for batch in tqdm_object:
                self.model.zero_grad()
                outputs = self.model(batch[0].to(self.device),
                    labels=batch[2].to(self.device), 
                    attention_mask = batch[1].to(self.device),
                    token_type_ids=None
                )
                # print(outputs)
                loss = outputs[0]  

                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_meter.update(loss.detach().item(), batch[0].shape[0])
                tqdm_object.set_postfix(train_loss=loss_meter.avg)
                # torch.cuda.empty_cache()
            val_loss = self.validate(val_loader)
            self.train_loss_list.append(loss.detach().item())
            self.val_loss_list.append(val_loss)

    def validate(self, val_loader):
        # ========================================
        #               Validation
        # ========================================
        self.model.eval()
        # Evaluate data for one epoch
        loss_meter = AverageMeter()
        tqdm_object = tqdm(val_loader, total=len(val_loader), desc='Validation')
        for batch in tqdm_object:   
            with torch.no_grad():
                outputs = self.model(batch[0].to(self.device),
                    labels=batch[2].to(self.device), 
                    attention_mask = batch[1].to(self.device),
                    token_type_ids=None
                )
                loss = outputs[0]  

            loss_meter.update(loss.detach().item(), batch[0].shape[0])
            tqdm_object.set_postfix(val_loss=loss_meter.avg)
        
        return loss_meter.avg
    
    def generate(self, text_lst, max_token=10, batch_size=10):
        # self.model.eval()
        print(1)
        outputs = []
        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = self.tokenizer(texts, truncation=True, padding = True, max_length=1024, return_tensors='pt')
            prompt = {key: value.to(self.device) for key, value in prompt.items()}
            outs = self.model.generate(**prompt, max_new_tokens=max_token, pad_token_id=self.tokenizer.eos_token_id, do_sample=False, early_stopping = True)
            outs = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
            outputs += outs
        return outputs


def test(texts, previous_token, end_token):
    y = [txt.split(end_token)[0].split(previous_token)[-1] for txt in texts]
    return y


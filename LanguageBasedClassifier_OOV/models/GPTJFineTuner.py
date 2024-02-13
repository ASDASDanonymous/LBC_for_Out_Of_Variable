import sys
sys.path.append('./')
sys.path.append('./../')
import openai, os, time, torch, sys, importlib, json, copy
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import numpy as np
from models import lora_gptj
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import transformers
from utils.helper import log
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer

# verbalizer vocab
yes_list = [3363, 3763, 10889, 826, 2081, 3376]
no_list = [1400, 645, 407, 10352, 1239, 8005]

def get_result(prob_tensor, y):
    y_pred = prob_tensor.argmax(dim=1)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    auc = roc_auc_score(y, prob_tensor[:, 1])  
    return acc, f1, auc


class GPTJFineTuner(object):
    def __init__(self, config:dict,train_jsonl,valid_jsonl,cuda_idx = 0):
        self.config = config
        self.train_jsonl=train_jsonl
        self.valid_jsonl=valid_jsonl
        self.device = torch.device('cuda:%d' % cuda_idx) if torch.cuda.is_available() else 'cpu'
        torch.cuda.set_device(cuda_idx)

        self.prob_list = []
    
    def init_model(self):
        print('=====Initialize a new GPTJ Model=====')
        print('======================================')
        self.ft_model = lora_gptj.LoRaQGPTJ(adapter=True, device=self.device)

    def fine_tune(self):
        self.init_model()

    def calculate_score(self, scores, main_index, synonym_indexes):
        main_score = scores[:, main_index]
        synonym_score = sum(scores[:, idx] for idx in synonym_indexes) / len(synonym_indexes)
        return 0.8 * main_score + 0.2 * synonym_score

    def generate(self, gpt, text_lst, max_token=10, batch_size=2):
    # gpt.model.eval()
        outputs = []
        final_probs = torch.tensor([])

        for i in np.arange(0, len(text_lst), batch_size):
            texts = text_lst[i:min(i + batch_size, len(text_lst))]
            prompt = gpt.tokenizer(texts, truncation=True, padding=True, max_length=1024, return_tensors='pt')
            prompt = {key: value.to(gpt.device) for key, value in prompt.items()}
            outs = gpt.model.generate(**prompt, output_attentions = True, output_scores = True, max_new_tokens=max_token, return_dict_in_generate=True, pad_token_id=gpt.tokenizer.eos_token_id, do_sample=False , early_stopping=True)
            input_length = 1 if gpt.model.config.is_encoder_decoder else prompt['input_ids'].shape[1]
            generated_tokens = outs.sequences[:, input_length]
            torch.set_printoptions(threshold=float('inf'))
            generated_token_ids = outs.sequences[:].tolist() 

            yes_score = self.calculate_score(outs.scores[0], yes_list[0], yes_list[1:])
            no_score = self.calculate_score(outs.scores[0], no_list[0], no_list[1:])

            batch_probs = F.softmax(torch.stack([no_score, yes_score], dim=1), dim=1)
            batch_probs = batch_probs.to('cpu')
            final_probs = torch.cat([final_probs,batch_probs], dim = 0)

            outs = gpt.tokenizer.batch_decode(outs.sequences, skip_special_tokens=True)
            outputs += outs
            
        return outputs, final_probs
    
    def prompt2value(self, x):
        results = x.strip().split('@@@')[0]
        return str(results)

    
    def query(self, gpt, prompts, bs=10):
        outputs = self.generate(gpt, prompts, batch_size=bs)
        ans = []
        prob_tensor_queryF = outputs[1]
        for txt in outputs[0]:
            try:
                output = self.prompt2value(txt.split('@@@')[0].split('###')[-1])
            except:
                output = None
            ans.append(output)
        print(f"ans = {ans}")
        return ans, prob_tensor_queryF


    def eval(self, valid_prompts, valid_df, test_prompts, test_df, logf, y_name='y', train_df = None):
        """
            number of valid samples
            L2 error on the valid samples
        """
        y_valid_outputs_, y_test_outputs_, len_valid_valid_y_, val_acc_list, test_acc_list, F1_score_list, auc_score_list = [], [], [], [], [], [], []
        best_idx = 0
        for model_idx in range(len(self.config['epochs'])):
            config = copy.deepcopy(self.config)
            epochs_ran = 0 if model_idx == 0 else self.config['epochs'][model_idx-1]
            config['epochs'] = self.config['epochs'][model_idx] - epochs_ran
            if self.new_model == True:
                print('==== Epoch %.4f ====' % self.config['epochs'][model_idx])
                self.ft_model.finetune(self.train_jsonl, 
                    self.valid_jsonl,
                    config,
                    '.../models/__pycache__')

            # validation
            valid_outputs = self.query(self.ft_model, valid_prompts, bs = 15)
            y_valid_outputs = valid_outputs[0]
            y_valid_outputs_.append(y_valid_outputs)
        
            valid_valid_y = [valid_df[y_name][i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]
            valid_valid_y_outputs = [y_valid_outputs[i] for i in range(len(y_valid_outputs)) if y_valid_outputs[i] != None]

            len_valid_valid_y = len(valid_valid_y)
            print("| Valid Val #outputs/Total #outputs:%d/%d" % (len_valid_valid_y,len(y_valid_outputs)))
            len_valid_valid_y_.append(len_valid_valid_y)

            val_prob_tensor = valid_outputs[1]
            val_acc, val_f1, val_auc = get_result(val_prob_tensor, valid_valid_y, True)
            val_acc_list.append(val_acc)

            print('| validation Acc     : %.2f' % (100*val_acc))
            print('| validation F1     : %.2f' % val_f1)
            print('| validation AUC     : %.2f' % val_auc)
            if (val_acc < val_acc_list[best_idx]) or (np.isnan(val_acc_list[best_idx])):
                best_idx = model_idx
            
            # Test
            test_outputs = self.query(self.ft_model, test_prompts, bs = 10)
            y_test_outputs = test_outputs[0]
            y_test_outputs_.append(y_test_outputs)

            valid_test_y = [test_df[y_name][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
            print("Test #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test_outputs)))
            test_prob_tensor = test_outputs[1]
            print(f"test_y : {valid_test_y}")
            test_acc, test_f1, test_auc = get_result(test_prob_tensor, valid_test_y, True)
            test_acc_list.append(test_acc)

            print('| Test Acc     : %.2f' % (100*test_acc))
            print('| F1 score     : %.2f' % test_f1)
            print('| AUC score     : %.2f' % test_auc)

        print('Selected epoch: %.4f' % self.config['epochs'][best_idx])
        self.best_idx = best_idx

        return y_test_outputs_[best_idx], len_valid_valid_y_,val_acc_list, test_acc_list


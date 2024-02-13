import os
import numpy as np
from functools import partial
import pandas as pd
import random
import sys

sys.path.append('')

def data2text_feature_name_Creditcard(row, train_quartiles, test_quartiles, mode='train'):
    quartiles = train_quartiles if mode == 'train' else test_quartiles

    prompt = ''

    if mode == 'test':
        prompt += "Additional Information: This applicant "

        if row[0] is not None:
            prompt += "is male, " if row[0] == 0 else "is female, "

        if row[3] is not None:
            prompt += "is not married, " if row[3] == 0 else "is married, "

        if row[4] is not None:
            prompt += "does not have a bank account, " if row[4] == 0 else "has a bank account, "

        if row[6] is not None:
            prompt += f"is of {row[6]} ethnicity, "

        employment_duration_quartile_1 = quartiles.iloc[0, 7]
        employment_duration_quartile_3 = quartiles.iloc[2, 7]
        if row[7] is not None:
            if row[7] <= employment_duration_quartile_1:
                prompt += "has been employed for a short duration, "
            elif row[7] <= employment_duration_quartile_3:
                prompt += "has been employed for a moderate duration, "
            else:
                prompt += "has been employed for a long duration, "

    prompt += "Trained Information: The applicant "

    age_quartile_1 = quartiles.iloc[0, 1]
    age_quartile_3 = quartiles.iloc[2, 1]
    if row[1] is not None:
        if row[1] <= age_quartile_1:
            prompt += "is of a young age, "
        elif row[1] <= age_quartile_3:
            prompt += "is of a middle age, "
        else:
            prompt += "is of an older age, "

    debt_quartile_1 = quartiles.iloc[0, 2]
    debt_quartile_3 = quartiles.iloc[2, 2]
    if row[2] is not None:
        if row[2] == 0:
            prompt += "has a zero debt, "
        elif row[2] <= debt_quartile_1:
            prompt += "has a low debt, "
        elif row[2] <= debt_quartile_3:
            prompt += "has a standard debt, "
        else:
            prompt += "has a high debt, "

    if row[5] is not None:
        prompt += f"works in the {row[5]} industry, "

    if row[9] is not None:
        prompt += "is currently employed. " if row[9] == 1 else "is not currently employed. "

    credit_score_quartile_1 = quartiles.iloc[0, 10]
    credit_score_quartile_3 = quartiles.iloc[2, 10]
    if row[10] is not None:
        if row[10] <= credit_score_quartile_1:
            prompt += "credit score is low, "
        elif row[10] <= credit_score_quartile_3:
            prompt += "credit score is standard, "
        else:
            prompt += "credit score is high, "

    if row[11] is not None:
        prompt += "does not have a driver's license, " if row[11] == 0 else "has a driver's license, "

    if row[12] is not None:
        if row[12] == 'c':
            prompt += "is a citizen, "
        elif row[12] == 'i':
            prompt += "is an immigrant, "
        elif row[12] == 'p':
            prompt += "is a foreigner, "

    if row[13] is not None:
        prompt += f"and resides in the zip code {row[13]}. "

    prompt += "Can the bank give this person a credit card?"

    completion = "Yes" if row['y'] == 1 else "No"
        
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)




def data2text_feature_name_Loan(row, train_quartiles, test_quartiles, mode='train'):
   
    quartiles = train_quartiles if mode == 'train' else test_quartiles

    prompt = ''

    if mode == 'test':
        prompt += "Additional Information: This person "
        
        if row[1] == "Male":
            prompt += "is male, "
        elif row[1] == "Female":
            prompt += "is female, "

        if row[2] == 'No':
            prompt += "is not married, "
        elif row[2] == 'Yes':
            prompt += "is married, "
        
        if row[4] == "Graduate":
            prompt += "graduated from university, "
        elif row[4] == "Not Graduate":
            prompt += "has not graduated from university, "
            
        if row[5] == "No":
            prompt += "is not self-employed, "
        elif row[5] == "Yes":
            prompt += "is self-employed, "

        if row[6] is not None:
            if row[6] <= quartiles.iloc[0, 6]:  
                prompt += "This person's income is considered low. "
            elif row[6] <= quartiles.iloc[2, 6]:
                prompt += "This person's income is considered standard. "
            else:
                prompt += "This person's income is considered high. "

    prompt += "Trained Information: this person "

    if row[3] in ['1','2']:    
        prompt += f"has {int(row[3])} dependents, "
    elif row[3] == '0':
        prompt += "has no dependents, "
    else:
        prompt += "has 3 or more dependents, "

    if row[7] is not None:
        if row[7] <= quartiles.iloc[0, 7]: 
            prompt += "This person's guardian earns a low salary. "
        elif row[7] <= quartiles.iloc[2, 7]:  
            prompt += "This person's guardian earns a standard salary. "
        else:
            prompt += "This person's guardian earns a high salary. "

    if row[8] is not None:
        if row[8] <= quartiles.iloc[0, 8]: 
            prompt += "The loan amount is low. "
        elif row[8] <= quartiles.iloc[2, 8]:
            prompt += "The loan amount is standard. "
        else:
            prompt += "The loan amount is high. "

    if row[9] is not None:
        prompt += f"this person's loan term is {int(row[9])} days, "
    if row[10] is not None:
        prompt += f"this person's credit history includes {int(row[10])} instances, "
    if row[11] is not None:
        prompt += f"this person's property area is {row[11]}. "
    
    prompt += "Should the bank give this person a loan? "

    completion = 'Yes' if str(row['y']) == '1' else "No"
        
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)



def data2text_feature_name_Steel_Plate(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles

    prompt = "This steel plates' information is as follows. "
    feature_names = 'X_Minimum	X_Maximum	Y_Minimum	Y_Maximum	Pixels_Areas	X_Perimeter	Y_Perimeter	Sum_of_Luminosity	Minimum_of_Luminosity	Maximum_of_Luminosity	Length_of_Conveyer	TypeOfSteel_A300	TypeOfSteel_A400	Steel_Plate_Thickness	Edges_Index	Empty_Index	Square_Index	Outside_X_Index	Edges_X_Index	Edges_Y_Index	Outside_Global_Index	LogOfAreas	Log_X_Index	Log_Y_Index	Orientation_Index	Luminosity_Index	SigmoidOfAreas	Pastry	Z_Scratch	K_Scatch	Stains	Dirtiness	Bumps'.split()
    
    #variable indexs : 0,1,2,3,7,8,9,10,14,15,19,20,21,22,28,30,31
    Out_of_variables = []
    In_variables = []
    shuffled_OOV = random.sample(Out_of_variables, len(Out_of_variables))
    
    if mode == 'train':
        new_order_cols = In_variables + ['y']
    else:
        new_order_cols = shuffled_OOV + In_variables + ['y']

        for col in enumerate(new_order_cols):
            if col == 'y':  
                continue
            if col == 0:
                prompt += 'Additional Information, '
            if feature_names[col] == feature_names[In_variables[0]]:  
                prompt += 'Trained Information, '
            feature_value = row[col]

            if categorical and feature_value is not None and col in quartiles.columns:

                if feature_value <= quartiles[col].loc[0.25]:
                    feature_value = "low"
                elif feature_value <= quartiles[col].loc[0.75]:
                    feature_value = "medium"
                else:
                    feature_value = "high"
            if feature_value is not None:
                prompt += f"{feature_names[col]} is {feature_value}, "
    
    prompt += "Is this steel plate defective?"

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)


def data2text_feature_name_Blood(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def data2text_feature_name_Breast_Cancer(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def data2text_feature_name_German(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def data2text_feature_name_ILPD(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def data2text_feature_name_Salary(row, cols, train_quartiles, test_quartiles, categorical, mode='train'):

    quartiles = train_quartiles if mode == 'train' else test_quartiles
    
    #write prompt components
    prompt = ''

    completion = "No" if row["y"] == 0 else "Yes"
    
    return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

def df2jsonl_feat_name(df, filename, did, train_quartiles, test_quartiles, integer = False):
    fpath = os.path.join('.../data', filename)

    if did == 'Blood':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Blood, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    
    if did == 'Breast_Cancer':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Breast_Cancer, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    
    if did == 'Creditcard':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Creditcard, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    
    elif did == 'German':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_German, cols=list(df.columns), train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath

    elif did == 'ILPD':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_ILPD, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath

    elif did == 'Loan':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Loan, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    
    elif did == 'Salary':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Salary, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath
    
    elif did == 'Steel_Plate':
        jsonl = '\n'.join(df.apply(func = partial(data2text_feature_name_Steel_Plate, cols=list(df.columns), categorical = True, train_quartiles = train_quartiles, test_quartiles = test_quartiles), axis = 1).tolist())
        with open(fpath, 'w') as f:
            f.write(jsonl)
        return fpath

    else:
        raise NotImplementedError
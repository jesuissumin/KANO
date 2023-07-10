import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
import csv

from chemprop.train.make_predictions import make_predictions
from chemprop.parsing import parse_predict_args

import numpy as np
from sklearn import metrics

def read_csv(fn, use_compound_names=False):
    with open(fn) as f:
        reader = csv.reader(f)
        next(reader)

        smiles_s = []
        label_s = []
        for line in reader:
            if use_compound_names:
                smiles = line[1]
                label = line[2]
            else:
                smiles = line[0]
                label = line[1]
            smiles_s.append(smiles)
            label_s.append(label)
    return smiles_s, label_s

def calc_acc(label_s, pred_s):
    pred_label_s = pred_s>0.5
    t = sum(pred_label_s==label_s)
    return t/len(label_s)

def calc_auroc(label_s, pred_s):
    auroc = metrics.roc_auc_score(label_s, pred_s)
    return auroc

def calc_confusion(label_s, pred_s):
    pred_label_s = pred_s>0.5
    tp = sum((pred_label_s==1)&(label_s==1))
    tn = sum((pred_label_s==0)&(label_s==0))
    fp = sum((pred_label_s==1)&(label_s==0))
    fn = sum((pred_label_s==0)&(label_s==1))
    return tp, tn, fp, fn

def calc_bac(label_s, pred_s):
    tp, tn, fp, fn = calc_confusion(label_s, pred_s)
    sen = tp/(tp+fn)
    spe = tn/(tn+fp)
    return (sen+spe)/2

def calc_f1(label_s, pred_s):
    tp, tn, fp, fn = calc_confusion(label_s, pred_s)
    if tp==0:
        return 0
    pre = tp/(tp+fp)
    rec = tp/(tp+fn)
    return 2*(pre*rec)/(pre+rec)

def calc_aupr(label_s, pred_s):
    aupr = metrics.average_precision_score(label_s, pred_s)
    return aupr

def calc_ece(label_s, pred_s):
    n_data = len(label_s)
    n_bins = 10
    bin_width = 1.0/n_bins
    ece = 0
    pred_label_s = pred_s>0.5
    acc_s = label_s==pred_label_s
    conf_s = np.array([pred_s[i] if pred_label_s[i]==1 else 1-pred_s[i] for i in range(len(pred_s))])
    # if p > 0.5, conf = p, else conf = 1-p
    for i in range(n_bins):
        lower_b = i*bin_width
        upper_b = (i+1)*bin_width
        if i < n_bins - 1:
            cond = (pred_s>=lower_b)&(pred_s<upper_b)
        else:
            cond = (pred_s>=lower_b)&(pred_s<=upper_b)
        if sum(cond)==0:
            continue
        acc = np.mean(acc_s[cond])
        conf = np.mean(conf_s[cond])
        ece += np.abs(acc-conf)*np.sum(cond)/n_data
    return ece


def calc_metrics(label_s, pred_s):
    label_s = np.array(label_s).astype(np.float)
    pred_s = np.squeeze(np.array(pred_s))
    
    acc = calc_acc(label_s, pred_s)
    auroc = calc_auroc(label_s, pred_s)
    bac = calc_bac(label_s, pred_s)
    f1 = calc_f1(label_s, pred_s)
    aupr = calc_aupr(label_s, pred_s)
    ece = calc_ece(label_s, pred_s)
    return [acc, auroc, bac, f1, aupr, ece]



if __name__=="__main__":
    args = parse_predict_args()
    args.use_compound_names = True

    # load dataset
    test_all_path = "/home/sumin/herg/data/BayeshERG/Finetuning/test_all.csv"
    test_rev_path = "/home/sumin/herg/data/BayeshERG/Finetuning/test_rev.csv"
    ext4_path = "/home/sumin/herg/data/BayeshERG/External/EX4.csv"

    test_all_smiles, test_all_label = read_csv(test_all_path, args.use_compound_names)
    test_rev_smiles, test_rev_label = read_csv(test_rev_path, args.use_compound_names)
    ext4_smiles, ext4_label = read_csv(ext4_path, args.use_compound_names)

    # make predictions
    test_all_pred, _ = make_predictions(args, test_all_smiles)
    test_all_result = ["test_all"] + calc_metrics(test_all_label, test_all_pred)
    
    test_rev_pred, _ = make_predictions(args, test_rev_smiles)
    test_rev_result = ["test_rev"] + calc_metrics(test_rev_label, test_rev_pred)
    
    ext4_pred, _ = make_predictions(args, ext4_smiles)
    ext4_result = ["ext4"] + calc_metrics(ext4_label, ext4_pred)

    
    # save the results as csv
    with open("herg_result.csv","w") as f:
        f.write(",".join(["set","acc","auc","bac","f1","aupr","ece"])+"\n")
        f.write(",".join(list(map(lambda x: f"{x:.3f}",test_all_result)))+"\n")
        f.write(",".join(list(map(lambda x: f"{x:.3f}",test_rev_result)))+"\n")
        f.write(",".join(list(map(lambda x: f"{x:.3f}",ext4_result)))+"\n")
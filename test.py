# encoding: utf-8
"""
Training implementation
"""
import os
import cv2
import csv
import math
import argparse
import numpy as np
import torch
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import GbDataSet, GbUsgDataSet
#from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, accuracy_score, confusion_matrix, f1_score
from collections import Counter
import json

from models import RadFormer
from PIL import Image
import pickle

import warnings
warnings.filterwarnings("ignore")

#np.set_printoptions(threshold = np.nan)


N_CLASSES = 3
CLASS_NAMES = ['nrml', 'benign', 'malg']


def parse():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--val_list', dest="val_list", default="data/cls_split/val.txt")
    parser.add_argument('--out_channels', dest="out_channels", default=2048, type=int)
    parser.add_argument('--model_file', dest="model_file", default="agcnn.pkl")
    parser.add_argument('--global_net', dest="global_net", default="resnet50")
    parser.add_argument('--local_net', dest="local_net", default="bagnet33")
    parser.add_argument('--global_weight', dest="global_weight", default=0.55, type=float)
    parser.add_argument('--local_weight', dest="local_weight", default=0.1, type=float)
    parser.add_argument('--fusion_weight', dest="fusion_weight", default=0.35, type=float)
    parser.add_argument('--num_layers', dest="num_layers", default=4, type=int)
    parser.add_argument('--score_file', dest="score_file", default="out_scores")
    parser.add_argument('--scheme', dest="scheme", default="majority")
    parser.add_argument('--pred_name', dest="pred_name", default="radformer.csv")
    args = parser.parse_args()
    return args


def main(args):
    #print('********************load data********************')
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    val_dataset = GbUsgDataSet(data_dir=args.img_dir, 
                            image_list_file=args.val_list,
                            transform=transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, num_workers=0)

    #print('********************load data succeed!********************')

    #print('********************load model********************')
    # initialize model
    
    model = RadFormer(local_net=args.local_net, \
                        global_weight=args.global_weight, \
                        local_weight=args.local_weight, \
                        fusion_weight=args.fusion_weight, \
                        use_rgb=True, num_layers=args.num_layers, pretrain=False).cuda()

    model.load_state_dict(torch.load(args.model_file)) 
    model.float().cuda()
   
    #print('********************load model succeed!********************')

    #print('*******validation*********')
    y_true, pred_g, pred_l, pred_f, patients, scores = validate(model, val_loader, args)
        
    #print_stats(y_true, pred_g, label="Global")
    #print_stats(y_true, pred_l, label="Local")
    print_stats(y_true, pred_f)

    d, y_true, y_pred, probs = get_patient_stats(patients, y_true, pred_f, 
                                    scores, scheme=args.scheme, name=args.pred_name)
    #jd = json.dumps(d)
    #print(jd)
    stats = get_stats(y_true, y_pred, probs)
    print("%.4f %.4f %.4f %.4f %.4f"%(stats["acc"], stats["spec"], stats["sens"], stats["auc"], stats["pr_auc"]))
    #print("%.4f"%stats["pr_auc"])
    c = stats["cfm"]
    print(c)

def get_patient_stats(patients, y_true, y_pred, y_score, scheme="max", name="pred.csv"):
    d = {}
    for i, p in enumerate(patients):
        if p not in d:
            tmp = {"gold": y_true[i], "pred": [], "score": []}
        else:
            tmp = d[p]
        preds = tmp["pred"]
        preds.append(y_pred[i])
        tmp["pred"] = preds
        scores = tmp["score"]
        scores.append(y_score[i])
        tmp["score"] = scores
        d[p] = tmp
    y_true, y_pred, probs = [], [], []
    res = []
    for p, v in d.items():
        if scheme=="max":
            prob = max(v["score"])
            pred = 2 if 2 in v["pred"] else 1
        elif scheme=="majority":
            count = Counter(v["pred"])
            pred = 2 if count[2] >= len(v["pred"])/3 and count[2]>0 else 1
            prob = count[2]/len(v["pred"]) if pred == 2 else (1 - (count[2]/len(v["pred"])))
            # sum_, len_ = 0, 0
            # for i, elem in enumerate(v["score"]):
            #     if v["pred"][i] == pred:
            #         sum_ += elem
            #         len_ += 1
            # prob = sum_/ len_ #sum(v["score"])/len(v["score"])
            #prob = sum(v["score"])/len(v["score"]) #sum_/ len_ 
        else:
            raise NotImplementedError("Patient level prediction scheme not implemented")
        v["final"] = pred
        v["proba"] = round(prob, 4)
        d[p] = v
        y_true.append(v["gold"])
        y_pred.append(v["final"])
        probs.append(v["proba"])
        res.append([p, v["gold"], v["final"], round(prob, 4)])
        
        sorted_res = sorted(res, key=lambda x: int(x[0]))
        with open(name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(sorted_res)

    return d, y_true, y_pred, probs

def get_stats(y_true, y_pred, probs):
    cfm = confusion_matrix(y_true, y_pred)
    if cfm.shape != (2,2):
        cfm = np.array([[0, 0], [0, cfm[0][0]]])
    acc = accuracy_score(y_true, y_pred)
    spec = cfm[0][0]/np.sum(cfm[0])
    sens = cfm[1][1]/np.sum(cfm[1])
    f1 = (2*spec*sens)/(spec+sens)
    
    pr, re, _ = precision_recall_curve(y_true, probs, pos_label=2)
    pr_auc = round(auc(re, pr), 4)

    try:
        auc_ = round(roc_auc_score(y_true, probs), 4)
    except:
        auc_ = np.nan
    
    res = { 
            "cfm": cfm,
            "acc": acc,
            "spec": spec,
            "sens": sens,
            "f1": round(f1, 3),
            "auc": auc_,
            "pr_auc": pr_auc
    }
    return res

def print_stats(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        cfm = confusion_matrix(y_true, y_pred)
        sens = cfm[2][2]/np.sum(cfm[2])
        spec = ((cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1])))
        acc_binary = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1] + cfm[2][2])/(np.sum(cfm))
        print("Acc-3class: %.4f Acc-Binary: %.4f Specificity: %.4f Sensitivity: %.4f"%(acc, acc_binary, spec, sens))
        print(cfm)

def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()
    #return pred.tolist()


def validate(model, val_loader, args):
    model.eval()
    y_true, pred_g, pred_l, pred_f = [], [], [], []
    score = []
    patients = []
    scores = []
    for i, (inp, target, fname) in enumerate(val_loader):
        with torch.no_grad():
            input_var = torch.autograd.Variable(inp.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            g_out, l_out, f_out, _ = model(input_var)

            patients.append(fname[0].split("/")[3])
            y_true.append(target.tolist()[0])
            pred_g.append(get_pred_label(g_out)) 
            pred_l.append(get_pred_label(l_out)) 
            pred_f.append(get_pred_label(f_out)) 
            scores.append(f_out.tolist()[0][pred_f[-1]])
            #score.append([y_true[-1], f_out.tolist()[0]])
            
    return y_true, pred_g, pred_l, pred_f, patients, scores


if __name__ == "__main__":
    args = parse()
    main(args)

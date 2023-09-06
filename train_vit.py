# encoding: utf-8
"""
Training implementation
"""
import os
import cv2
import argparse
import numpy as np
import torch
import json
torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataloader import GbDataSet, GbUsgDataSet
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from models import ViT, DEiT, SwinT
from PIL import Image

#import neptune.new as neptune

#np.set_printoptions(threshold = np.nan)


N_CLASSES = 3
CLASS_NAMES = ['nrml', 'benign', 'malg']


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--train_list', dest="train_list", default="data/cls_split/train.txt")
    parser.add_argument('--val_list', dest="val_list", default="data/cls_split/val.txt")
    parser.add_argument('--meta_file', dest="meta_file", default="data/res.json")
    parser.add_argument('--out_channels', dest="out_channels", default=2048, type=int)
    parser.add_argument('--epochs', dest="epochs", default=30, type=int)
    parser.add_argument('--save_dir', dest="save_dir", default="expt")
    parser.add_argument('--save_name', dest="save_name", default="attnbag")
    parser.add_argument('--batch_size', dest="batch_size", default=16, type=int)
    parser.add_argument('--lr', dest="lr", default=0.0001, type=float)
    parser.add_argument('--optim', dest="optim", default="adam")
    parser.add_argument('--num_layers', dest="num_layers", default=2, type=int)
    parser.add_argument('--arch', dest="arch", default="vit")
    args = parser.parse_args()
    return args


def main(args):
    print('********************load data********************')

    device = torch.device("cuda")
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    with open(args.meta_file, "r") as f:
        df = json.load(f)

    train_dataset = GbUsgDataSet(data_dir=args.img_dir,
                            image_list_file=args.train_list,
                            #df=df,
                            #train=True,
                            transform=transforms.Compose([
                                #transforms.Resize((224,224)),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=0)
    
    val_dataset = GbUsgDataSet(data_dir=args.img_dir, 
                            image_list_file=args.val_list,
                            #df=df,
                            #train=True,
                            transform=transforms.Compose([
                                #transforms.Resize((224,224)),
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize,
                            ]))

    val_loader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, num_workers=0)

    print('********************load data succeed!********************')


    print('********************load model********************')
    # initialize model
    if args.arch == "vit":
        model = ViT().to(device)
    elif args.arch == "deit":
        model = DEiT().to(device)
    elif args.arch == "swin":
        model = SwinT().to(device)
    else:
        raise(NotImplementedError("Architecture not supported"))
    
    criterion = nn.CrossEntropyLoss().to(device)
    #cudnn.benchmark = False
   
    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    
    lr_sched = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 1)
    print('********************load model succeed!********************')

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    save_model_name = args.save_name
    best_acc = 0
    best_ep = 0

    print('********************begin training!********************')
    for epoch in range(args.epochs):
        #print('Epoch {}/{}'.format(epoch , args.epochs - 1))
        #print('-' * 10)
        #set the mode of model
        model.train()  #set model to training mode
        
        total_acc_train = 0
        total_loss_train = 0.0

        for i, (train_image, train_label, fnames) in enumerate(train_loader):
            output = model(train_image.to(device))
            loss = criterion(output, train_label.to(device))
            acc = (output.argmax(dim=1) == train_label.to(device)).sum().item()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        acc, spec, sens = validate(model, val_loader)

        print(f'Epochs: {epoch + 1} | Loss: {total_loss_train / len(train_dataset): .3f} | Accuracy: {total_acc_train / len(train_dataset): .3f} | Val Acc: {acc: .4f} | Val Spec: {spec: .4f} | Val Sens: {sens: .4f}')

        #save
        torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
        #if best_acc < acc: #max(acc_g, acc_l, acc_f):
        #    best_acc= acc 
        #    #best_ep = epoch
        #    torch.save(model.state_dict(), save_path+"/"+save_model_name+'_epoch_'+str(epoch)+'.pkl')
        #    print('Best acc model saved!')

        # LR schedular step
        lr_sched.step()  #about lr and gamma


def get_pred_label(pred_tensor):
    _, pred = torch.max(pred_tensor, dim=1)
    return pred.item()


def validate(model, val_loader):
    model.eval()
    y_true, y_pred = [], []
    for i, (global_inp, target, filenames) in enumerate(val_loader):
        with torch.no_grad():
            global_input_var = torch.autograd.Variable(global_inp.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            
            outs = model(global_input_var)

            pred = get_pred_label(outs)
            
            y_true.append(target.tolist()[0])
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = np.sum(cfm[:2,:2])/np.sum(cfm[:2])
    sens = cfm[2][2]/np.sum(cfm[2])
    return acc, spec, sens

if __name__ == "__main__":
    args = parse()
    main(args)

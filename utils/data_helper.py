import os
import numpy as np
import torch.utils.data
import torch
from PIL import Image
import json
import random
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import h5py

import models

def process_input(in_list_str):
    in_list = in_list_str.strip().split()
    out_list = []
    for word in in_list:
        for w in word:
            out_list.append(w)
    out_list_str = " ".join(out_list)
    return out_list_str

class BiDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, img_path, txt_out_path, value_to_idx, split, dataset):
        self.json_path = json_path
        self.txt_out_path = txt_out_path
        self.img_path = img_path
        self.split = split
        self.dataset = dataset
        self.original_src = []
        self.original_tgts = []
        self.img_name = []
        self.input_img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
        ])

        self.ori_attrs = []
        self.img_out = []
        with open(self.txt_out_path, "r") as f:
            for index, line in enumerate(f):
                line = line.strip() 
                line_list = line.split()

                value_one_hot = torch.zeros(len(value_to_idx)) # multi_hot for words in line
                for word in line_list:
                    if word in value_to_idx:
                        value_one_hot[value_to_idx[word]] = 1
                self.ori_attrs.append(value_one_hot)

                if len(line) > 0:
                    tmp = " ".join(line_list)
                    self.img_out.append(tmp)
                else:
                    self.img_out.append(" ")



        if split == 'train':
            count_line = 0
            with open(self.json_path, "r") as f:
                jsonf = json.load(f)
                for i, name in enumerate(jsonf):
                    for j in range(len(jsonf[name]['tgt'])):
                        self.img_name.append(name)
                        self.original_src.append(
                            process_input(jsonf[name]['src'])
                        )
                        self.original_tgts.append(process_input(jsonf[name]['tgt'][j]))
                        count_line += 1
        else:
            count_line = 0
            with open(self.json_path, "r") as f:
                jsonf = json.load(f)
                for i,name in enumerate(jsonf):
                    self.img_name.append(name)
                    self.original_src.append(
                        process_input(jsonf[name]['src'])
                    )
                    self.original_tgts.append(
                        [process_input(jsonf[name]['tgt'][j]) for j in range(len(jsonf[name]['tgt']))]
                    )
                    count_line += len(jsonf[name]['tgt'])


    def __getitem__(self, index):
        out_dict = {}

        img_id = self.img_name[index]
        out_dict['img_id'] = img_id

        img = Image.open(
            os.path.join(self.img_path, f"{img_id}.jpg")
        ).convert("RGB")
        out_dict['simg'] = self.input_img_transform(img)

        out_dict['split'] = self.split
        out_dict['dataset'] = self.dataset

        out_dict["ori_src"] = self.original_src[index]
        out_dict["ori_tgt"] = self.original_tgts[index]
        out_dict["ori_attrs"] = self.ori_attrs[index] # multi-hot vector
        out_dict['attr_names'] = self.img_out[index]

        return out_dict

    def __len__(self):
        return len(self.img_name)


def collate_fn(data):
    batch_entry = {}

    B = len(data)

    fname = os.path.join(
        '../CEPSUM_dataset/jdsum', data[0]['dataset'], "frcnn_{}_{}.h5".format(data[0]['dataset'], data[0]['split']) 
    )
    f = h5py.File(fname, 'r')
    batch = []
    for i in range(B):
        out_dict = data[i]

        img_id = out_dict['img_id']
        
        try:
            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]
            n_boxes = len(boxes)
        except Exception as e:
            n_boxes = 0

        if n_boxes > 0 :
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            # np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            # np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)
            boxes.clamp_(min=0.0, max=1.0)

            feats = np.zeros(shape=(n_boxes, 2048), dtype=np.float32)
            f[f'{img_id}/vis_feas'].read_direct(feats)
            feats = torch.from_numpy(feats)

            vis_attention_mask = torch.tensor([1] * n_boxes + [0] * (36 - n_boxes), dtype=torch.float32)

            obj_cls = np.zeros(shape=(n_boxes, 1601), dtype=np.float32)
            f[f'{img_id}/obj_cls'].read_direct(obj_cls)
            obj_cls = torch.from_numpy(obj_cls)

            out_dict['vis_feas'] = feats
            out_dict['boxes'] = boxes
            out_dict['vis_attention_mask'] = vis_attention_mask
            out_dict['obj_cls'] = obj_cls
            out_dict['n_boxes'] = n_boxes
        else:
            out_dict['vis_feas'] = torch.zeros([1, 2048], dtype=torch.float32)
            out_dict['boxes'] = torch.zeros([1, 4], dtype=torch.float32)
            out_dict['vis_attention_mask'] = torch.ones([1], dtype=torch.float32)
            out_dict['obj_cls'] = torch.zeros([1, 1601], dtype=torch.float32)
            out_dict['n_boxes'] = 1
        
        batch.append(out_dict)


    V_L = max(entry['n_boxes'] for entry in batch)
    feat_dim = batch[0]['vis_feas'].shape[-1]

    boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
    vis_feats = torch.zeros(B, V_L, feat_dim, dtype=torch.float)
    vis_attention_mask = torch.zeros(B, V_L, dtype=torch.float)
    obj_cls = torch.zeros(B, V_L, 1601, dtype=torch.float)

    img_len = []
    ori_src = []
    ori_tgt = []
    ori_attrs = []
    attr_names = []
    simg = torch.zeros((B, ) + batch[0]['simg'].shape)
    for i, entry in enumerate(batch):
        n_boxes = entry['n_boxes']
        img_len.append(n_boxes)
        boxes[i, :n_boxes] = entry['boxes']
        vis_feats[i, :n_boxes] = entry['vis_feas']
        vis_attention_mask[i, :n_boxes] = 1
        obj_cls[i, :n_boxes] = entry['obj_cls']
        simg[i] = entry['simg']

        ori_src.append(entry['ori_src'])
        ori_tgt.append(entry['ori_tgt'])
        ori_attrs.append(entry['ori_attrs'])

        if entry['attr_names'] is not None:
            attr_names.append(entry['attr_names'])

    
    batch_entry['boxes'] = boxes
    batch_entry['vis_feats'] = vis_feats
    batch_entry['vis_attention_mask'] = vis_attention_mask
    batch_entry['obj_cls'] = obj_cls
    batch_entry['n_boxes'] = img_len

    batch_entry['ori_src'] = ori_src
    batch_entry['ori_tgt'] = ori_tgt
    batch_entry['ori_attrs'] = ori_attrs
    batch_entry['simg'] = simg
    batch_entry['attr_names'] = attr_names

    return batch_entry

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def tokenize_input(text, tokenizer, max_len=None):
    if max_len == None:
        max_len = max([len(sent) for sent in text])
    tokens = tokenizer.batch_encode_plus(text, max_length=max_len, pad_to_max_length=True, truncation=True, return_tensors='pt', add_special_tokens=False)
    return tokens['input_ids'], tokens['attention_mask']

def tokenize_output(texts, tokenizer, max_len):
    all_tokens = []
    all_mask = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokens[:max_len-1] + [tokenizer.eos_token_id] 
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [0] * (max_len - len(tokens))
        all_tokens.append(tokens)
        all_mask.append(mask)

    return torch.tensor(all_tokens), torch.tensor(all_mask)

def token_tgtv2(data_tgt):
    '''
    tgt2tensor
    '''
    tgt_tensor = torch.zeros((len(data_tgt),) + data_tgt[0].shape, dtype=torch.float16)
    for i,s in enumerate(data_tgt):
        tgt_tensor[i] = data_tgt[i]
    return tgt_tensor

def Hausdorff_dist(X, Y):
    len_X, len_Y = X.shape[0], Y.shape[0]
    X = torch.nn.functional.normalize(X, dim=2)
    Y = torch.nn.functional.normalize(Y, dim=2)
    X = X.unsqueeze(1).expand(-1, len_Y, -1, -1)
    Y = Y.unsqueeze(0).expand(len_X, -1, -1, -1)
    dist = torch.sqrt(torch.sum((X - Y) ** 2, dim=3))
    d_X_Y = dist.min(dim=1)[0].max(dim=0)[0]
    d_Y_X = dist.min(dim=0)[0].max(dim=0)[0]
    hd = torch.stack([d_X_Y, d_Y_X], dim=1).max(dim=1, keepdim=True)[0]
    return hd
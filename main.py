import os
from numpy import *
import warnings
from torch import optim, einsum
import time
import datetime
import torch.nn.functional as F
import torch.nn as nn
from transformers import BertTokenizer
import torch
import argparse
import json

import utils
import models

from models.modeling_bart import VLBart


torch.autograd.set_detect_anomaly(True)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_num_threads(4)
warnings.filterwarnings("ignore")

LIMIT_NUM = 5

ROOT_PATH = "../CEPSUM_dataset/jdsum"

tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")

idx_to_value = {}
value_to_idx = {}

def get_args():
    parser = argparse.ArgumentParser(description='part2')
    parser.add_argument("--root", default="../CEPSUM_dataset/jdsum", type=str, help="root path")
    parser.add_argument('--gpu', default="0", type=str, help="Use CUDA on the device.")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_beams', default=3, type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', default='train', type=str, help="Mode selection")
    parser.add_argument('--lr', default='3e-5', type=float, help="learning rate")
    parser.add_argument('--epoch', default='5', type=int, help="train epoch")
    parser.add_argument('--seed', default='2022', type=int, help="train seed")

    parser.add_argument('--ReS_lambda', default='0.05', type=float, help="loss lambda")
    parser.add_argument('--mi_t', default='1', type=float, help="train MI temperature")
    parser.add_argument('--clip_gamma', default='5', type=float, help="loss gamma")

    parser.add_argument('--confidence', default='0.2', type=float, help="theta. threshold of final predicted attributes.")

    parser.add_argument('--dataset', default='cases_bags', type=str, help="")

    args = parser.parse_args()
    return args

def shift_tokens_right(input_ids, token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  #index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = torch.ones_like(prev_output_tokens[:, 0], device=prev_output_tokens.device) * token_id
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens

class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)


class MyModelv3(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        model = VLBart.from_pretrained("fnlp/bart-base-chinese", num_classes=config.num_classes) # see all BART models at https://huggingface.co/models?filter=bart
        self.bart = model
        self.config = self.bart.config

        self.mrm_head = BartClassificationHead(
            self.config.d_model,
            self.config.d_model,
            1601,
            0.1,
        )
        self._init_weights(self.mrm_head.dense)
        self._init_weights(self.mrm_head.out_proj)

        self.vis2text = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.ReLU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self._init_weights(self.vis2text[0])
        self._init_weights(self.vis2text[2])

        self.vis_cls_norm = LayerNorm(self.config.d_model)
        self.text_cls_norm = LayerNorm(self.config.d_model)
        self.img_to_latents = EmbedToLatents(self.config.d_model, self.config.d_model)
        self.text_to_latents = EmbedToLatents(self.config.d_model, self.config.d_model)
        self._init_weights(self.img_to_latents.to_latents)
        self._init_weights(self.text_to_latents.to_latents)

        # contrastive learning temperature
        self.temperature = nn.Parameter(torch.Tensor([1.]))

    def _init_weights(self, module):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, attrs, attrs_att, decoder_ids, decoder_attention_mask, image_features, vis_attention_mask, img_len):
        decoder_input_ids = shift_tokens_right(decoder_ids, self.config.bos_token_id)

        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        vis_cls_embeds, visual_encoder_outputs = self.bart.model.vis_enc(image_features, img_len)

        logits, loss, attr_logits, hidden_state = self.bart(
            input_ids=input_ids,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=decoder_ids,
            reduce_loss=True,
            vis_attention_mask=vis_attention_mask,
            visual_encoder_outputs=visual_encoder_outputs,
            return_dict=False,
        )
        return loss,\
                encoder_outputs[0], visual_encoder_outputs,\
                vis_cls_embeds, attr_logits


    def generate_text(self, input_ids, attention_mask, attrs, attrs_att, image_features, vis_attention_mask, img_len, beam_size=10):
        encoder_outputs = self.bart.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        generate_ids = self.bart.generate(
            input_ids=None,
            attention_mask=attention_mask,
            use_cache=True,
            decoder_start_token_id=self.config.bos_token_id, 
            num_beams=beam_size, 
            max_length=90,
            early_stopping=True,
            encoder_outputs=encoder_outputs,
            vis_inputs=image_features,
            vis_attention_mask=vis_attention_mask,
            img_len=img_len,
        )

        ans = [ tokenizer.decode(w) for w in generate_ids]
        
        for i, s_a in enumerate(ans):
            if len(s_a) <= 0:
                ans[i] = ' '
            else:
                s_a = s_a.replace(tokenizer.bos_token, '')
                s_a = s_a.replace(tokenizer.eos_token, "")
                s_a = s_a.replace(tokenizer.sep_token, "")

                s_a = s_a.replace('[', "")
                s_a = s_a.replace(']', "")
                s_a = s_a.replace('EOS', "")
                ans[i] = s_a
        
        return ans


    def mrm(self, vis_inputs, vis_attention_mask, img_len, box_cls, mrm_probability=0.15):
        vis_feas, boxes = vis_inputs
        bsz, num_objs, _ = vis_feas.shape
        device = vis_feas.device
        vis_attention_mask = vis_attention_mask.bool()
        probability_matrix = torch.full([bsz, num_objs],
                                        mrm_probability,
                                        dtype=torch.float, device=device)
        masked_regions = torch.bernoulli(probability_matrix).bool()
        mrm_labels = []
        mrm_masks = torch.zeros([bsz, num_objs], device=device)
        mrm_masks[masked_regions & vis_attention_mask] = 1
        vis_embeds = self.bart.model.vis_enc.visual_embedding(vis_feas, boxes, None, None)
        image_embeds = vis_embeds.clone()
        for i in range(bsz):
            masked_indices = masked_regions[i][vis_attention_mask[i]].nonzero(as_tuple=False)
            mrm_labels.append(box_cls[i][masked_indices].clone())
            image_embeds[i][masked_indices] = torch.zeros(
                (len(masked_indices), 1, self.config.d_model),
                dtype=image_embeds[i].dtype, device=device)
        
        visual_encoder_outputs = self.bart.model.vis_enc.img_transformer(image_embeds, img_len)[-1]

        region_representation = visual_encoder_outputs[mrm_masks.bool()] # [bsz, num_reg, dim]
        if len(region_representation) > 0:
            predict_cls = self.mrm_head(region_representation) # [bsz, num_reg, num_classes]
            predict_cls = F.log_softmax(predict_cls, dim=-1)
            mrm_labels = torch.cat(mrm_labels,
                                   dim=0).to(visual_encoder_outputs.device) # [bsz, num_reg]
            mrm_loss = F.kl_div(predict_cls.double(),
                                mrm_labels.double().squeeze(1),
                                reduction='batchmean')
        else:
            mrm_loss = 0

        return mrm_loss
   

def build_model(checkpoints, config):
    
    """
    build model
    :param checkpoints: load checkpoint if there is pretrained model
    :return: model, optimizer and the print function
    """

    # model
    print("building model...\n")
    model_txt = MyModelv3(config)
    model_txt.to(config.device)
    if checkpoints is not None:
        model_txt.load_state_dict(checkpoints["model_txt"])


    optimizer_txt = optim.Adam(filter(lambda p: p.requires_grad, model_txt.parameters()), lr=config.lr)
    param_count_txt = sum([param.contiguous().view(-1).size()[0] for param in model_txt.parameters()])
    print("total number of model txt parameters: %d\n\n" % param_count_txt)

    return model_txt, optimizer_txt,


def train_model(model_txt, data, optimz_txt, params, config):
    device = config.device
    model_txt.to(device)
    model_txt.train()
    train_loader = data["train_loader"]

    patience = best_model_iter = 0
    eval_updates = len(train_loader) // 10

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimz_txt, T_max=config.epoch)

    attr_loss = nn.BCEWithLogitsLoss()


    for i in range(1, config.epoch + 1):
        print("now epoch: ", i)
        for step_i, batch in enumerate(train_loader):
            input_img_feas = (batch['vis_feats'].to(device), batch['boxes'].to(device))
            vis_attention_mask = batch['vis_attention_mask'].to(device)
            obj_cls = batch['obj_cls'].to(device)
            img_len = batch['n_boxes']

            ori_src = batch['ori_src']
            ori_tgt = batch['ori_tgt']
            ori_attrs = batch['ori_attrs']

            src, src_att = utils.tokenize_input(ori_src, tokenizer, 400) # add bos and eos token
            tgt, tgt_att = utils.tokenize_output(ori_tgt, tokenizer, 90)

            attrs = utils.token_tgtv2(ori_attrs)
            attrs = attrs.to(device)
            attrs_att=None

            src, tgt = src.to(device), tgt.to(device)
            src_att, tgt_att = src_att.to(device), tgt_att.to(device)

            # (batch_size, tgt_sent_len, tgt_vocab_size)
            loss_ce,\
            textual_encoder_outputs, visual_encoder_outputs, vis_cls_embeds,\
            attr_logits = model_txt(
                src, 
                src_att, 
                attrs, 
                attrs_att,
                tgt, 
                tgt_att, 
                input_img_feas, 
                vis_attention_mask,
                img_len, 
            )
            text_cls_embeds = textual_encoder_outputs[:, -1]
            textual_encoder_outputs = textual_encoder_outputs[:, :-1]

            # attribute prediction
            loss_attr = attr_loss(attr_logits, attrs)

            # ITCL
            text_latents = model_txt.text_to_latents(text_cls_embeds)
            image_latents = model_txt.img_to_latents(vis_cls_embeds)

            sim = einsum('i d, j d -> i j', text_latents, image_latents)
            sim = sim * model_txt.temperature.exp()
            contrastive_labels = torch.arange(text_latents.shape[0], device=device)

            ce = F.cross_entropy
            loss_cl = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5

            # MRM
            loss_mrm = model_txt.mrm(input_img_feas, vis_attention_mask, img_len, obj_cls)

            # Hd
            hd_img_feas = model_txt.vis2text(visual_encoder_outputs)
            
            hausdorff_dist = utils.Hausdorff_dist(textual_encoder_outputs.transpose(1, 0), hd_img_feas.transpose(1, 0))
            loss_hd = (hausdorff_dist ** 2).mean()

            loss = loss_ce + 0.3*(loss_attr+ loss_hd) + 0.05*loss_cl + 0.8* loss_mrm 


            optimz_txt.zero_grad()
            loss.backward()
            optimz_txt.step()
            scheduler.step()

            params["updates"] += 1


            if params["updates"]%eval_updates==0:
                ## report
                print("train: loss_ce({})".format(loss_ce.item()))
                print("train: loss_cl({})".format(loss_cl.item()))
                print("train: loss_mrm({})".format(loss_mrm.item()))
                print("train: loss_hd({})".format(loss_hd.item()))
                print("train: loss_attr({})".format(loss_attr.item()))
                ## evaluation
                print("evaluating after %d updates...\r" % params["updates"])
                print("--valid--")
                score = eval_model(model_txt, data, config, params)
                
                if score < params['metric'][-1]:
                    patience += 1
                    print('hit patience %d' % patience)
                if patience >= 5:
                    print('early stop!')
                    print('the best model is from iteration [%d]' % best_model_iter,)
                    return
                
                params['metric'].append(score)
                if score >= max(params['metric']):
                    best_model_iter = params['updates']
                    save_model(
                        params['model_path'],
                        model_txt,
                        optimz_txt,
                        params["updates"],
                        config,
                    )
                model_txt.train()


def test_model(model_txt, data, config, params):
    device = config.device
    model_txt.to(device)
    model_txt.eval()
    reference, candidate = [], []
    test_loader = data['test_loader']
    num_beams = config.num_beams
    print('num_beams:', num_beams)
    
    for step_i, batch in enumerate(test_loader):
        with torch.no_grad():
            vis_feats = batch['vis_feats']
            vis_attention_mask = batch['vis_attention_mask']
            boxes = batch['boxes']
            img_len = batch['n_boxes']

            bs = vis_feats.shape[0]
            expanded_return_idx = (
                torch.arange(bs).view(-1, 1).repeat(1, num_beams).view(-1)
            )
            vis_feats = vis_feats.index_select(0, expanded_return_idx).to(device)
            boxes = boxes.index_select(0, expanded_return_idx).to(device)
            vis_attention_mask = vis_attention_mask.index_select(0, expanded_return_idx).to(device)
            expand_len = [img_len[i] for i in expanded_return_idx]

            input_img_feas = (vis_feats, boxes)

            ori_src = batch['ori_src']
            ori_tgt = batch['ori_tgt']

            ori_attrs = batch['ori_attrs']
            attrs = utils.token_tgtv2(ori_attrs)
            attrs = attrs.to(device)
            attrs_att=None

            src, src_att = utils.tokenize_input(ori_src, tokenizer, 400)

            src = src.to(device)
            src_att = src_att.to(device)

            samples = model_txt.generate_text(
                src, src_att, 
                attrs, attrs_att, 
                input_img_feas, vis_attention_mask, expand_len, 
                beam_size=num_beams,
            )
            
        candidate += [sample.split() for sample in samples]
        tmp = []
        for ss in ori_tgt:
            tmpp = []
            for s in ss:
                tmpp.append(s.split())
            tmp.append(tmpp)
        reference += tmp



    score = utils.srouge(
        reference, candidate, device, print
    )
    return score


def eval_model(model_txt, data, config, params):
    device = config.device
    model_txt.to(device)
    model_txt.eval()
    reference, candidate = [], []
    valid_loader = data["valid_loader"]

    # num_beams = config.num_beams
    num_beams = 1

    for step_i, batch in enumerate(valid_loader):
        with torch.no_grad():
            vis_feats = batch['vis_feats']
            vis_attention_mask = batch['vis_attention_mask']
            boxes = batch['boxes']
            img_len = batch['n_boxes']

            bs = vis_feats.shape[0]
            expanded_return_idx = (
                torch.arange(bs).view(-1, 1).repeat(1, num_beams).view(-1)
            )
            vis_feats = vis_feats.index_select(0, expanded_return_idx).to(device)
            boxes = boxes.index_select(0, expanded_return_idx).to(device)
            vis_attention_mask = vis_attention_mask.index_select(0, expanded_return_idx).to(device)
            expand_len = [img_len[i] for i in expanded_return_idx]

            input_img_feas = (vis_feats, boxes)

            ori_src = batch['ori_src']
            ori_tgt = batch['ori_tgt']
            ori_attrs = batch['ori_attrs']
            attrs = utils.token_tgtv2(ori_attrs)
            attrs = attrs.to(device)
            attrs_att=None

            src, src_att = utils.tokenize_input(ori_src, tokenizer, 400)

            src = src.to(device)
            src_att = src_att.to(device)

            samples = model_txt.generate_text(
                src, src_att, 
                attrs, attrs_att, 
                input_img_feas, vis_attention_mask, expand_len, 
                beam_size=num_beams
            )

        candidate += [sample.split() for sample in samples]
        tmp = []
        for ss in ori_tgt:
            tmpp = []
            for s in ss:
                tmpp.append(s.split())
            tmp.append(tmpp)
        reference += tmp 

    score = utils.srouge_eval(
        reference, candidate, print
    )
    return score # ROUGE-2

# save model
def save_model(path, model_txt, optim_txt, updates, config):
    model_txt_state_dict = model_txt.state_dict()
    optim_txt_state_dict = optim_txt.state_dict()
    checkpoints = {
        "model_txt": model_txt_state_dict,
        "config": config,
        "updates": updates,
        "optim_txt": optim_txt_state_dict,
    }
    torch.save(checkpoints, path)


def main():
    time1_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    print("start: ", time1_str)
    config = get_args()

    utils.set_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    config.device = torch.device("cuda:{}".format(config.gpu))
    config.root = ROOT_PATH

    global idx_to_value, value_to_idx
    limit_num_dict = {"cases_bags": 1000, "clothing": 10000, "home_appliances": 5000}
    vocab_file = os.path.join(config.root, config.dataset, "new_word_dict_{}.json".format(limit_num_dict[config.dataset]))
    with open(vocab_file, 'r') as f:
        value_to_idx = json.load(f)
        for value in value_to_idx:
            idx_to_value[value_to_idx[value]] = value
    print("num_classes: ", len(idx_to_value)) # vocab size of attribute words
    config.num_classes = len(idx_to_value)

    params = {
        "updates": 0,
        "report_total": 0,
        "log_path": os.path.join("exp", "{}".format(config.dataset)) + "/",
        'model_path': os.path.join("exp", "{}".format(config.dataset), "best_checkpoint.pt"),
        "best_model_path": "",
    }
    os.makedirs(params['log_path'], exist_ok=True)

    checkpoints = None
    # model_path = params["model_path"]
    # checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_txt, optimizer_txt = build_model(checkpoints, config)

    st_time = time.time()
    print("loading dataset...") 
    train_set = utils.BiDataset(
        os.path.join(config.root, config.dataset, "{}_train.json".format(config.dataset)), 
        os.path.join(config.root, config.dataset, "img"),
        os.path.join(config.root, config.dataset, "train_ext_tgt_output.txt"), 
        value_to_idx, 'train', config.dataset)
    test_set = utils.BiDataset(
        os.path.join(config.root, config.dataset, "{}_test.json".format(config.dataset)), 
        os.path.join(config.root, config.dataset, "img"),
        os.path.join(config.root, config.dataset, "test_ext_tgt_output.txt"), 
        value_to_idx, 'test', config.dataset)
    valid_set = utils.BiDataset(
        os.path.join(config.root, config.dataset, "{}_dev.json".format(config.dataset)),
        os.path.join(config.root, config.dataset, "img"), 
        os.path.join(config.root, config.dataset, "dev_ext_tgt_output.txt"), 
        value_to_idx, 'dev', config.dataset)



    train_loader = torch.utils.data.DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers * 4, 
        collate_fn=utils.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers, 
        collate_fn=utils.collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=config.num_workers,
        collate_fn=utils.collate_fn
    )
    data = {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "valid_loader": valid_loader,
    }

    print("cost {}s\n".format(time.time()-st_time))
    print(f"len(train_loader):{len(data['train_loader'])}")
    print(f"len(valid_loader):{len(data['valid_loader'])}")
    print(f"len(test_loader):{len(data['test_loader'])}")


    params['metric'] = [0]

    if config.mode == "train":
        print("train:\n")
        train_model(model_txt, data, optimizer_txt, params, config)

        print("Best score: {}\n".format(max(params['metric'])))
        model_path = params["model_path"]
        checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_txt.load_state_dict(checkpoints['model_txt'])
        model_txt = model_txt.to(config.device)
        score = test_model(model_txt, data, config, params)
        print(f"Best score: {score}")       
    elif config.mode == "test":
        print("test:\n")
        model_path = params["model_path"]
        checkpoints = torch.load(model_path, map_location=lambda storage, loc: storage)
        print('updates:', checkpoints['updates'])
        model_txt.load_state_dict(checkpoints['model_txt'])
        model_txt = model_txt.to(config.device)
        score = test_model(model_txt, data, config, params)
        print(f"Best score: {score}") 
    else:
        score = eval_model(model_txt, data, config, params)

    time2_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S')
    print("over: ", time2_str)


if __name__=='__main__':
    print("\n\n=============\n\n")
    main()
    print("\n\n=============\n\n")

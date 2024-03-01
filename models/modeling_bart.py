
import math
import random
from dataclasses import dataclass

from transformers.models.bart.modeling_bart import (
    BartLearnedPositionalEmbedding,
    BartEncoderLayer, BartEncoder, BartDecoder,
    BartPretrainedModel, BartModel, BartForConditionalGeneration,
    BartConfig,
    ACT2FN,
    shift_tokens_right, _make_causal_mask, _expand_mask
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import copy

from transformers.modeling_outputs import ModelOutput, BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from transformers import BeamScorer, BeamSearchScorer

from models.img_transformer import ImageTransformerEncoder

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from einops import rearrange, repeat

logger = logging.get_logger(__name__)
# Bart + VisEnc

class VisualEmbedding(nn.Module):
    def __init__(self, config, obj_order_embedding):
        super().__init__()
        self.config = config
        # feat_dim = config.feat_dim
        # pos_dim = config.pos_dim
        # # n_objs = config.n_objs
        # n_images = config.n_images
        feat_dim = 2048
        pos_dim = 4
        # n_objs = config.n_objs
        n_images = 1

        if self.config.individual_vis_layer_norm:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            if self.config.use_vis_layer_norm:
                feat_embedding.append(nn.LayerNorm(config.d_model))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            if self.config.use_vis_layer_norm:
                absolute_vis_pos_embedding.append(nn.LayerNorm(config.d_model))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

                # self.default_obj_order_ids = self.config.default_obj_order_ids
        else:
            # Object feature encoding
            feat_embedding = [nn.Linear(feat_dim, config.d_model)]
            self.feat_embedding = nn.Sequential(*feat_embedding)

            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, config.d_model)]
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)

            if self.config.use_vis_order_embedding:
                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, config.d_model)

                # self.default_obj_order_ids = self.config.default_obj_order_ids

            if self.config.use_vis_layer_norm:
                self.layer_norm = nn.LayerNorm(config.d_model)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, :, 3] - pos[:, :, 2]
        width = pos[:, :, 1] - pos[:, :, 0]
        area = height * width
        return area

    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        B, N, _ = feats.size()
        assert pos.size() == (B, N, 4)

        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(2)  # [B, N, 1]
        pos = torch.cat([pos, area], dim=2)  # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)

        if self.config.use_vis_order_embedding:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0)  # .expand(B, -1)
            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0)
            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)

            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding + obj_order_embedding

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

        if not self.config.individual_vis_layer_norm:
            if self.config.use_vis_layer_norm:
                vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding

class VisualEncoder(nn.Module):
    def __init__(self, config, vis_embedding):
        super().__init__()

        config.individual_vis_layer_norm = True
        config.use_vis_order_embedding = True
        config.use_vis_layer_norm = True
        visual_feature_dim = 2048
        self.img_transformer = ImageTransformerEncoder(d_model=config.d_model, num_layers=4, num_heads=8, dim_feedforward=2048)
        self.visual_embedding = VisualEmbedding(config, vis_embedding)
        self.vis_proj = nn.Linear(visual_feature_dim, config.d_model)
        self.vis_cls_token = nn.Parameter(torch.randn(config.d_model))

    def forward(self, vis_inputs, img_len):
        vis_feats = vis_inputs[0]
        boxes = vis_inputs[1]
        img_order_ids = None
        obj_order_ids = None
        if len(vis_inputs) >= 3:
            img_order_ids = vis_inputs[2]
        if len(vis_inputs) == 4:
            obj_order_ids = vis_inputs[3]

        vis_embeds = self.visual_embedding(vis_feats, boxes, img_order_ids, obj_order_ids)
        
        vis_cls_tokens = repeat(self.vis_cls_token, 'd -> b 1 d', b=vis_embeds.shape[0])
        vis_embeds = torch.cat((vis_cls_tokens, vis_embeds), dim=-2)
        img_len = [i + 1 for i in img_len]

        visual_encoder_outputs = self.img_transformer(vis_embeds, img_len)[-1]

        return visual_encoder_outputs[:, 0], visual_encoder_outputs[:, 1:]


class JointEncoder(BartEncoder):
    """
    BartEncoder + attr-text attention
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

        self.config = config

        self.text_cls_token = nn.Parameter(torch.randn(config.d_model))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,

        attr=None,

        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        ############################ modify
        batch = inputs_embeds.shape[0]
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        inputs_embeds = torch.cat((inputs_embeds, text_cls_tokens), dim=-2)

        cls_mask = torch.ones(batch, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat((attention_mask, cls_mask[:, None]), dim=1)
        ###################################
        embed_pos = self.embed_positions(inputs_embeds)

        inputs_embeds = inputs_embeds + embed_pos


        inputs_embeds = self.layernorm_embedding(inputs_embeds)


        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)


        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask, layer_head_mask=None, output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class VLBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super(BartModel, self).__init__(config)
        self.dropout = config.dropout

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # self.encoder = BartEncoder(config, self.shared)
        #----- Modified-----#
        self.encoder = JointEncoder(config, self.shared)
        self.vis_enc = VisualEncoder(config, self.shared)

        dim_common = 256
        n_attn_heads = 1
        self._linear_1 = nn.Linear(config.d_model, dim_common) # K
        self._linear_2 = nn.Linear(config.d_model, dim_common) # V
        self._linear_3 = nn.Linear(config.d_model, dim_common) # Q
        self.mm = nn.Linear(dim_common, dim_common)
        self._multi_head_attn = nn.MultiheadAttention(dim_common, n_attn_heads)
        self._linear_4 = nn.Linear(config.d_model + dim_common, config.d_model)
        self.fg = nn.Linear(dim_common + config.d_model, dim_common)
        self.sigmiod = nn.Sigmoid()

        self.l1 = nn.Linear(config.d_model, dim_common, bias=False)
        self.l2 = nn.Linear(config.d_model, dim_common, bias=False)
        self.l3 = nn.Linear(config.d_model, dim_common, bias=False)
        self.l4 = nn.Linear(dim_common, config.num_classes)
        #-------------------#
        
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

        self.visual_embedding.obj_order_embedding = self.shared

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,
        img_len=None,
        visual_encoder_outputs=None,

        attr=None,
        get_final_encoder_outputs=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        **kwargs,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        
        textual_encoder_outputs = encoder_outputs[0][:, :-1]
        text_cls_embeds = encoder_outputs[0][:, -1]

        # visual encoder
        if visual_encoder_outputs is None:
            _, visual_encoder_outputs = self.vis_enc(vis_inputs, img_len)

        # text-vision fusion
        K = self._linear_1(visual_encoder_outputs).transpose(0, 1)
        V = self._linear_2(visual_encoder_outputs).transpose(0, 1)
        Q = self._linear_3(textual_encoder_outputs).transpose(0, 1)
        attn_output, _ = self._multi_head_attn(Q, K, V, key_padding_mask=(1 - vis_attention_mask).bool())
        attn_output = attn_output.transpose(0, 1)

        forget_mask = self.fg(torch.cat((textual_encoder_outputs, attn_output), 2))
        forget_mask = self.sigmiod(forget_mask)
        forget_mask = F.dropout(forget_mask, p=self.dropout, training=self.training)
        attn_output = forget_mask.mul(self.mm(attn_output))
        
        final_encoder_outputs = self._linear_4(torch.cat((textual_encoder_outputs, attn_output), 2)) + textual_encoder_outputs

        if get_final_encoder_outputs:
            return final_encoder_outputs

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=torch.float, device=input_ids.device)
        encoder_attention_mask = attention_mask

        if attr:
            # pred attrs
            txt_context = textual_encoder_outputs * encoder_attention_mask.unsqueeze(2)
            txt_context = F.dropout(self.l1(torch.sum(txt_context, dim=1)), p=self.dropout, training=self.training)
            mm_context = final_encoder_outputs * encoder_attention_mask.unsqueeze(2)
            mm_context = F.dropout(self.l2(torch.sum(mm_context, dim=1)), p=self.dropout, training=self.training)
            text_cls_embeds = F.dropout(self.l3(text_cls_embeds), p=self.dropout, training=self.training)
            attr_logits = self.l4(txt_context + mm_context + text_cls_embeds)
            attr_preds = torch.sigmoid(attr_logits)

        else:
            attr_logits = None

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=final_encoder_outputs,
            # encoder_attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return (decoder_outputs[0], attr_logits)

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class VLBart(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig, num_classes):
        super(BartForConditionalGeneration, self).__init__(config)
        config.num_classes = num_classes
        self.model = VLBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,

        vis_inputs=None,
        vis_attention_mask=None,
        img_len=None,
        visual_encoder_outputs=None,

        attr=None,

        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        reduce_loss=False,

        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if not return_dict:
            hidden_state, attr_logits = self.model(
                input_ids,
                attention_mask=attention_mask,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,
                img_len=img_len,
                visual_encoder_outputs=visual_encoder_outputs,

                attr=True,

                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(hidden_state) + self.final_logits_bias
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,

                vis_inputs=vis_inputs,
                vis_attention_mask=vis_attention_mask,
                img_len=img_len,
                visual_encoder_outputs=visual_encoder_outputs,

                attr=True,

                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            if reduce_loss:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction='none')
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # output = (lm_logits,) + outputs[1:]
            # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            return (lm_logits, masked_lm_loss, attr_logits, hidden_state)

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        output = {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

        if "vis_attention_mask" in kwargs:
            output["vis_attention_mask"] = kwargs['vis_attention_mask']

        if "vis_inputs" in kwargs:
            output["vis_inputs"] = kwargs['vis_inputs']

        if "img_len" in kwargs:
            output["img_len"] = kwargs['img_len']

        if "attr" in kwargs:
            output["attr"] = kwargs['attr']

        return output

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1,
                                                                expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(
                0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx)

        if model_kwargs.get("vis_attention_mask", None) is not None:
            model_kwargs['vis_attention_mask'] = model_kwargs['vis_attention_mask'].index_select(
                0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs

        return input_ids, model_kwargs

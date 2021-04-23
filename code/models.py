import math
import os

from transformers import ElectraModel, ElectraPreTrainedModel, ElectraConfig, PreTrainedModel
import transformers
import torch
import torch.nn as nn



class ElectraRelationClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, loc1, loc2, **kwargs):
        x1 = features[range(len(loc1)), loc1, :] # (batch_size, hidden_state_dim)
        x2 = features[range(len(loc2)), loc2, :] # (batch_size, hidden_state_dim)
        x_total = torch.cat([x1, x2], dim=1) # (batch_size, hidden_state_dim * 2)

        x = self.dropout(x_total)
        x = self.dense(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x



class ElectraForSemanticAnalysis(ElectraPreTrainedModel):
    def __init__(self, config, resize_vocab_num=-1, pretrained_name=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = transformers.ElectraModel(config)
        if pretrained_name:
            self.model = transformers.ElectraModel(config).from_pretrained(pretrained_name)
        else:
            self.model = transformers.ElectraModel(config)
        if resize_vocab_num > 0:
            self.model.resize_token_embeddings(resize_vocab_num)
            print("Resized!!")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = ElectraRelationClassificationHead(self.config)

        self.init_weights()

    
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None, 
                inputs_embeds=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                # return_dict=None,
                entity1_ids=None, # 내가 추가한 인자
                entity2_ids=None, # 내가 추가한 인자
                labels=None
                ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        sequence_output = outputs[0] # last_hidden_states을 가져와줌

        logits = self.classifier(sequence_output, entity1_ids, entity2_ids)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[1:]
        #     return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }


class ElectraRelationClassificationHead2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        ent1s = []
        cls_loc = (features == 0).nonzero()
        sep1_loc = (features == 2).nonzero()
        x1 = features[range(len(loc1)), loc1, :] # (batch_size, hidden_state_dim)
        x2 = features[range(len(loc2)), loc2, :] # (batch_size, hidden_state_dim)
        x_total = torch.cat([x1, x2], dim=1) # (batch_size, hidden_state_dim * 2)

        # x = self.dropout(x_total)
        x = self.dense(x)
        x = torch.nn.functional.gelu(x)
        # x = self.dropout(x)
        x = self.out_proj(x)

        return x



class ElectraForSemanticAnalysis2(XLMRobertaPreTrainedModel):
    def __init__(self, config, resize_vocab_num=-1, pretrained_name=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = transformers.ElectraModel(config)
        if pretrained_name:
            self.model = transformers.ElectraModel(config).from_pretrained(pretrained_name)
        else:
            self.model = transformers.ElectraModel(config)
        if resize_vocab_num > 0:
            self.model.resize_token_embeddings(resize_vocab_num)
            print("Resized!!")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = ElectraRelationClassificationHead(self.config)

        self.init_weights()
    
    def forward(self, 
                input_ids=None, 
                attention_mask=None, 
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None, 
                inputs_embeds=None, 
                output_attentions=None, 
                output_hidden_states=None,
                labels=None
                ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        sequence_output = outputs[0] # last_hidden_states을 가져와줌

        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }



if __name__ =="__main__":
    config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    config.num_labels = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ElectraForSemanticAnalysis.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config, resize_vocab_num = 1)
    # model.to(device)
    print("ElectraForSemanticAnalysis model is working well")
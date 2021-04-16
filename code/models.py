import math
import os

from transformers import ElectraModel, ElectraPreTrainedModel
import transformers
import torch
import torch.nn as nn

# def get_model(model_name, model_structure):
#     model_list = {

#                 }

#     type_list = {

#     }
#     if model_name in model_list and model_structure in model_structure:


class ElectraRelationClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, loc1, loc2, **kwargs):
        x1 = features[range(len(loc1)), loc1, :] # (batch_size, hidden_state_dim)
        x2 = features[range(len(loc2)), loc2, :] # (batch_size, hidden_state_dim)
        x_total = torch.cat(x1, x2, dim=1) # (batch_size, hidden_state_dim * 2)

        x = self.dropout(x_total)
        x = self.dense(x)
        x = get_activation("gelu")(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class ElectraForSemanticAnalysis(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.model = transformers.ElectraModel(config)
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
                labels=None, 
                output_attentions=None, 
                output_hidden_states=None, 
                return_dict=None,
                entity_ids=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        print(outputs.shape)
        sequence_output = outputs[0] # last_hidden_states을 가져와줌

        # entity_ids: shape (batch_size, 2) 각 데이터 별로 entity1 토큰
        if not entity_ids:
            id1 = id2 = [0 * len(input_ids)]
        else:
            id1, id2 = entity_ids[:, 0], entity_ids[:, 1]

        logits = self.classifier(sequence_output, id1, id2)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits, 
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions
        )



if __name__ =="__main__":
    config = transformers.ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")
    config.num_classes = 42
    model = ElectraForSemanticAnalysis.from_pretrained("monologg/koelectra-base-v3-discriminator", config=config)
    print("ElectraForSemanticAnalysis model is working well")
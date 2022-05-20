import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
)

class HeadlineClassifier(nn.Module):
    def __init__(
        self, pretrained_model_name: str, num_classes: int = 2, dropout: float = 0.5,
        mean_pool: bool=True
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes
        )
        for k in config.to_dict().keys():
          if 'dropout' in k and 'classifier' not in k:
            config.update({k:0.3})

        self.model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(dropout)
        self.mean_pool=mean_pool
        self.num_labels = num_classes
        self.softmax = nn.Softmax(dim=-1)
        # if torch.cuda.is_available():
        #   self.model.cuda()
        #   self.classifier.cuda()
        #   self.dropout.cuda()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        # print('I am in forward method')
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print('I got output')
        hidden_state = output[0]  # (bs, seq_len, dim)
        outputs = (hidden_state,)
        if not self.mean_pool:
          pooled_output = hidden_state[:, 0]  # (bs, dim)
        else:
          pooled_output = hidden_state.mean(axis=1)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        probs = self.softmax(logits)
        predicted_label = torch.argmax(logits)
        predicted_prob = torch.max(probs)
        return predicted_label,predicted_prob,probs.detach().numpy()
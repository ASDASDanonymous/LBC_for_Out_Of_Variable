#####
#Add this file at 
#.../transformers/models/gptj/modeling_gptj.py
#####

import torch
import torch.nn as nn
import torch.nn.functional as F

class SynonymAwareCrossEntropyLoss(nn.Module):
    def __init__(self, tokenizer):
        super(SynonymAwareCrossEntropyLoss, self).__init__()
        self.tokenizer = tokenizer
        # Create a dictionary to map tokens to their synonym token IDs
        self.synonym_token_ids = {3363: [3763, 10889, 826, 2081, 3376], 1400: [645, 407, 10352, 1239, 8005], 1029: [3334], 7090: {13398}, 1877: {7754}, 6045: {1239,4844}, 10073: {1790,1449}, 5882:{890}, 1097:{1097}, 5719:{5719}, 1323:{1323}}
        self.base_loss_fct = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        # Flatten the logits and labels
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)

        # Compute the base loss for each element
        base_losses = self.base_loss_fct(flat_logits, flat_labels)

        # Prepare tensors for efficient computation
        device = flat_logits.device
        labels_tensor = torch.tensor(range(flat_logits.size(-1)), device=device)
        for i, label in enumerate(flat_labels):
            if label == -100:  # Skip -100 labels
                continue

            if label.item() in self.synonym_token_ids:
                # Vectorized computation for synonyms
                syn_ids = self.synonym_token_ids[label.item()]
                syn_labels = torch.tensor(list(syn_ids), device=device)
                syn_logits = flat_logits[i].repeat(len(list(syn_ids)), 1)
                # Calculate losses for all synonyms at once
                syn_losses = self.base_loss_fct(syn_logits, syn_labels) * (0.2 / len(list(syn_ids)))

                # Compute the total loss for the current label and its synonyms
                base_loss = base_losses[i] * 0.8
                total_loss = base_loss + syn_losses.sum()
                base_losses[i] = total_loss

        # Filter out the losses for ignored indices (-100)
        valid_losses = base_losses[flat_labels != -100]

        return valid_losses.mean()
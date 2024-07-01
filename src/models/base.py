from torch import nn
import torch
from transformers import AutoModel


class Base(nn.Module):
    def __init__(self, config, encoder_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            config["encoder_name"], output_hidden_states=True
        )
        self.learned_layers = list(range(config["nb_frozen_layers"] + 1, 12))
        self.drop = nn.Dropout(config["encoder_dropout"])
        for name, param in self.encoder.named_parameters():
            if (
                len(
                    [
                        layer_index
                        for layer_index in self.learned_layers
                        if str(layer_index) in name
                    ]
                )
                == 0
            ):
                param.requires_grad = False

    def forward(self, token_ids, token_map, len_samples, subtoken_map, mask_ids):
        """Produce an embedding for each sub-word input."""
        output = self.encoder(token_ids, attention_mask=mask_ids)
        attention_outputs = output[-1:][0][-2:]
        concat_outputs = torch.cat(attention_outputs, -1)
        kept_outputs = []
        for len_sample, output_index in zip(
            len_samples, range(concat_outputs.shape[0])
        ):
            kept_outputs.append(concat_outputs[output_index, 1 : len_sample + 1])

        aggregated_outputs = torch.cat(kept_outputs, 0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        final_output = torch.zeros(
            (aggregated_outputs.shape), dtype=aggregated_outputs.dtype, device="cuda:0"
        ).scatter_reduce_(
            0,
            subtoken_map,
            aggregated_outputs,
            reduce="mean",
            include_self=False,
        )
        final_output = torch.index_select(final_output, 0, token_map)
        return self.drop(final_output)

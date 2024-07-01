import torch
from models.nlp import FeedForward


class PRUNER(torch.nn.Module):
    def __init__(self, config: dict, max_span_len: int, batch_size: int):
        super().__init__()

        self.linear_output = torch.nn.Linear(384, 1)
        self.max_size = max_span_len
        self.batch_size = batch_size
        self.bilstm = torch.nn.LSTM(
            1536,
            config["bilstm_hidden_size"],
            bidirectional=True,
            num_layers=config["bilstm_nb_layers"],
            dropout=config["bilstm_dropout"],
        )
        self.ffnn_biaffine = FeedForward(4 * config["bilstm_hidden_size"], config)

    def apply_pruning(self, input):
        begin_indices = [
            index
            for index in range(input.shape[0])
            for _ in range(min(input.shape[0] - index, self.max_size))
        ]
        end_indices = [
            higher_index
            for index in range(input.shape[0])
            for higher_index in range(index, min(input.shape[0], index + self.max_size))
        ]
        final_end_indices = [
            end_index
            for index in range(input.shape[0])
            for end_index in range(min(input.shape[0] - index, self.max_size))
        ]
        output_tensor = torch.zeros(
            input.shape[0],
            self.max_size,
            1,
            device="cuda:0",
            dtype=input.dtype,
        )
        b = input[begin_indices, :]
        e = input[end_indices, :]
        bilinear_outputs = self.linear_output(
            torch.relu(
                self.ffnn_biaffine(
                    torch.cat(
                        (
                            b,
                            e,
                        ),
                        -1,
                    )
                )
            )
        )
        output_tensor[begin_indices, final_end_indices] = bilinear_outputs
        return output_tensor

    def forward(self, subword_embeddings):
        """_summary_

        Args:
            subword_embeddings (_type_): _description_
        """
        embeddings, _ = self.bilstm(subword_embeddings)
        output_tensor = self.apply_pruning(embeddings)
        return output_tensor

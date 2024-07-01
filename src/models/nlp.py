"""NLP models."""

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim_input, config):
        super(FeedForward, self).__init__()
        self.dim_output = dim_input
        self.layers = []
        self.create_ffnn(config)
        self.layers = nn.Sequential(*self.layers)

    def create_ffnn(self, config):
        for dim in config["ffnn_dims"]:
            self.layers.append(nn.Linear(self.dim_output, dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(config["ffnn_dropout"]))
            self.dim_output = dim

    def forward(self, tensor):
        return self.layers(tensor)


class NLP(torch.nn.Module):
    def __init__(
        self, config: dict, max_span_len: int, batch_size: int, output_len: int
    ):
        super().__init__()

        self.linear_output = torch.nn.Linear(config["ffnn_dims"][-1], output_len)
        self.output_len = output_len
        self.max_size = max_span_len
        self.batch_size = batch_size
        self.ffnn = FeedForward(8 * config["bilstm_hidden_size"], config)

    def get_predictions(self, input, begin_indexes, end_indexes):
        b_left_indexes = [
            index for index in begin_indexes for _ in range(len(begin_indexes))
        ]
        b_right_indexes = [
            index for _ in range(len(begin_indexes)) for index in begin_indexes
        ]
        e_left_indexes = [
            index for index in end_indexes for _ in range(len(begin_indexes))
        ]
        e_right_indexes = [
            index for _ in range(len(begin_indexes)) for index in end_indexes
        ]

        final_mention_begin = [
            index
            for index in range(len(begin_indexes))
            for _ in range(len(begin_indexes))
        ]
        final_mention_end = [
            index
            for _ in range(len(begin_indexes))
            for index in range(len(begin_indexes))
        ]
        output_tensor = torch.zeros(
            len(begin_indexes),
            len(begin_indexes),
            self.output_len,
            device="cuda:0",
            dtype=input.dtype,
        )
        b_1 = input[b_left_indexes, :]
        e_1 = input[e_left_indexes, :]
        b_2 = input[b_right_indexes, :]
        e_2 = input[e_right_indexes, :]
        bilinear_outputs = self.linear_output(
            torch.relu(
                self.ffnn(
                    torch.cat(
                        (
                            b_1,
                            e_1,
                            b_2,
                            e_2,
                        ),
                        -1,
                    )
                )
            )
        )

        output_tensor[final_mention_begin, final_mention_end] = bilinear_outputs.float()
        return output_tensor

    def forward(self, subword_embeddings, begin_indexes, end_indexes):
        """_summary_

        Args:
            subword_embeddings (_type_): _description_
        """
        output_tensor = self.get_predictions(
            subword_embeddings, begin_indexes, end_indexes
        )
        return output_tensor

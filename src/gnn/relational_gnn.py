import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import RGCNConv


class RelationalGNN(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        num_relations,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(
            RGCNConv(
                input_channels,
                hidden_channels,
                num_relations
            )
        )

        for _ in range(
            num_layers - 2
        ):

            self.convs.append(
                RGCNConv(
                    hidden_channels,
                    hidden_channels,
                    num_relations
                )
            )

        self.convs.append(
            RGCNConv(
                hidden_channels,
                output_channels,
                num_relations
            )
        )

    def forward(
        self,
        x,
        edge_index,
        edge_type
    ):

        for index, conv in enumerate(
            self.convs
        ):

            x = conv(
                x,
                edge_index,
                edge_type
            )

            if index != len(
                self.convs
            ) - 1:

                x = F.relu(x)

                x = F.dropout(
                    x,
                    p=self.dropout,
                    training=self.training
                )

        return x


if __name__ == "__main__":

    model = RelationalGNN(
        input_channels=11,
        hidden_channels=128,
        output_channels=64,
        num_relations=3,
        num_layers=3,
        dropout=0.2
    )

    print()
    print(
        "========================================"
    )
    print(
        "        RELATIONAL GNN MODEL"
    )
    print(
        "========================================"
    )

    print()
    print(
        model
    )

    print()
    print(
        "Input features: 11"
    )

    print(
        "Hidden channels: 128"
    )

    print(
        "Output embedding size: 64"
    )

    print(
        "Relationship types: 3"
    )

    print(
        "GNN layers: 3"
    )

    print(
        "Dropout: 0.2"
    )

    print()
    print(
        "Model created successfully."
    )
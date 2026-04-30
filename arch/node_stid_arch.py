import torch
from torch import nn

from .mlp import MultiLayerPerceptron


class NodeSTID(nn.Module):
    """
    Node-wise STID: A modified version of STID that trains on individual nodes
    instead of all nodes simultaneously.

    Original Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline
                    for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233

    Key Difference: Instead of processing all N nodes at once (shape [B, L, N, C]),
    this model processes one node at a time (shape [B, L, 1, C]).
    The spatial (node) embedding is removed since each sample is a single node.
    Only temporal embeddings (time-of-day, day-of-week) are kept.
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args.get("if_T_i_D", True)
        self.if_day_in_week = model_args.get("if_D_i_W", True)

        # NO spatial/node embeddings — each sample is a single node

        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer: flatten input_len * input_dim into embed_dim
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # encoding hidden dim = embed_dim + temporal dims
        self.hidden_dim = (
            self.embed_dim
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(self.num_layer)]
        )

        # regression layer
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of NodeSTID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, 1, C]
                Each sample is a single node's time series.

        Returns:
            torch.Tensor: prediction with shape [B, output_len, 1, 1]
        """

        # prepare data: select input features
        input_data = history_data[..., range(self.input_dim)]  # [B, L, 1, C]

        # temporal embeddings
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]  # [B, L, 1]
            # Use last time step's time-of-day to look up embedding
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]  # [B, 1, temp_dim_tid]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]  # [B, L, 1]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]  # [B, 1, temp_dim_diw]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape  # num_nodes = 1
        input_data = input_data.transpose(1, 2).contiguous()          # [B, 1, L, C]
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)  # [B, L*C, 1, 1]
        time_series_emb = self.time_series_emb_layer(input_data)      # [B, embed_dim, 1, 1]

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_tid, 1, 1]
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))  # [B, temp_dim_diw, 1, 1]

        # concatenate all embeddings (no node embedding)
        hidden = torch.cat([time_series_emb] + tem_emb, dim=1)  # [B, hidden_dim, 1, 1]

        # encoding
        hidden = self.encoder(hidden)  # [B, hidden_dim, 1, 1]

        # regression
        prediction = self.regression_layer(hidden)  # [B, output_len, 1, 1]

        return prediction

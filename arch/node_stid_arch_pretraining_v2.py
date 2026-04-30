import torch
from torch import nn

from .mlp import MultiLayerPerceptron


class NodeSTIDv2(nn.Module):
    """
    Node-wise STID v2 for pretraining on combined datasets.

    Extends NodeSTID with a third learnable metadata embedding: **dataset ID**.
    - Time-of-day embedding (TOD)
    - Day-of-week embedding (DOW)
    - Dataset ID embedding (which dataset this node belongs to)

    The dataset ID is passed as an extra channel (index 3) in the input data.
    The time series embedding still uses channels [0, 1, 2] (flow, TOD, DOW)
    via input_dim=3. The dataset ID channel is read separately to look up
    the learnable dataset embedding.

    Input shape: [B, L, 1, C] where C >= 4
        channel 0: flow (target)
        channel 1: time_of_day (normalized 0-1)
        channel 2: day_of_week (normalized 0-1)
        channel 3: dataset_id (integer 0, 1, 2, ...)
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]       # 3 (flow, tod, dow) for time series embedding
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args.get("if_T_i_D", True)
        self.if_day_in_week = model_args.get("if_D_i_W", True)
        self.if_dataset_id = model_args.get("if_dataset_id", True)

        # Dataset ID embedding
        self.num_datasets = model_args.get("num_datasets", 4)
        self.dataset_id_dim = model_args.get("dataset_id_dim", 64)
        if self.if_dataset_id:
            self.dataset_id_emb = nn.Parameter(
                torch.empty(self.num_datasets, self.dataset_id_dim))
            nn.init.xavier_uniform_(self.dataset_id_emb)

        # Temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # Embedding layer: flatten input_len * input_dim into embed_dim
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # Hidden dim = embed_dim + temporal dims + dataset_id_dim
        self.hidden_dim = (
            self.embed_dim
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
            + self.dataset_id_dim * int(self.if_dataset_id)
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(self.num_layer)]
        )

        # Regression layer
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True
        )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of NodeSTIDv2.

        Args:
            history_data (torch.Tensor): shape [B, L, 1, C] where C >= 4.
                Channels: [flow, time_of_day, day_of_week, dataset_id]

        Returns:
            torch.Tensor: prediction with shape [B, output_len, 1, 1]
        """

        # ---- Time series features (first input_dim channels) ----
        input_data = history_data[..., :self.input_dim]  # [B, L, 1, 3]

        # ---- Temporal embeddings ----
        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]  # [B, L, 1]
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

        # ---- Dataset ID embedding ----
        if self.if_dataset_id:
            # dataset_id is channel 3, constant across time steps; use last step
            ds_id_data = history_data[..., 3]  # [B, L, 1]
            ds_ids = ds_id_data[:, -1, :].long()  # [B, 1]
            dataset_emb = self.dataset_id_emb[ds_ids]  # [B, 1, dataset_id_dim]
        else:
            dataset_emb = None

        # ---- Time series embedding ----
        batch_size, _, num_nodes, _ = input_data.shape  # num_nodes = 1
        input_data = input_data.transpose(1, 2).contiguous()          # [B, 1, L, 3]
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)  # [B, L*3, 1, 1]
        time_series_emb = self.time_series_emb_layer(input_data)      # [B, embed_dim, 1, 1]

        # ---- Collect all embeddings ----
        embeddings = [time_series_emb]

        if time_in_day_emb is not None:
            embeddings.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))   # [B, tid, 1, 1]
        if day_in_week_emb is not None:
            embeddings.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))   # [B, diw, 1, 1]
        if dataset_emb is not None:
            embeddings.append(dataset_emb.transpose(1, 2).unsqueeze(-1))       # [B, ds_dim, 1, 1]

        hidden = torch.cat(embeddings, dim=1)  # [B, hidden_dim, 1, 1]

        # ---- Encoding ----
        hidden = self.encoder(hidden)  # [B, hidden_dim, 1, 1]

        # ---- Regression ----
        prediction = self.regression_layer(hidden)  # [B, output_len, 1, 1]

        return prediction

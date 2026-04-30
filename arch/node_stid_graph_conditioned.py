import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MultiLayerPerceptron


class GraphConvLayer(nn.Module):
    """
    Diffusion-style graph convolution (from Graph WaveNet).
    Supports multiple support matrices and higher-order diffusion.
    """

    def __init__(self, c_in, c_out, support_len=2, order=2, dropout=0.1):
        super().__init__()
        self.order = order
        self.support_len = support_len
        # Input channels = original + (order * support_len) hop features
        total_c_in = c_in * (order * support_len + 1)
        self.fc = nn.Conv2d(total_c_in, c_out, kernel_size=(1, 1), bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, supports):
        """
        Args:
            x: [B, C, N, 1]
            supports: list of [N, N] adjacency matrices

        Returns:
            [B, c_out, N, 1]
        """
        out = [x]
        for a in supports:
            a = a.to(x.device)
            x1 = torch.einsum('bcni,nm->bcmi', x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('bcni,nm->bcmi', x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.fc(h)
        h = self.dropout(h)
        return h


class GraphConditioningBlock(nn.Module):
    """
    A block that applies graph convolution + MLP with residual connection.
    """

    def __init__(self, hidden_dim, support_len=2, order=2, dropout=0.1):
        super().__init__()
        self.gcn = GraphConvLayer(hidden_dim, hidden_dim, support_len, order, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, supports):
        """
        Args:
            x: [B, C, N, 1]
            supports: list of adjacency matrices [N, N]
        Returns:
            [B, C, N, 1]
        """
        # Graph convolution with residual
        residual = x
        x = self.gcn(x, supports)
        x = x + residual
        # LayerNorm: permute to [B, N, 1, C] for norm over C
        x = x.permute(0, 2, 3, 1)  # [B, N, 1, C]
        x = self.norm(x)
        # MLP with residual
        residual2 = x
        x = self.mlp(x)
        x = x + residual2
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, N, 1]
        return x


class NodeSTIDGraphConditioned(nn.Module):
    """
    Final architecture: Pretrained NodeSTIDv2 backbone + Graph Conditioning.

    Pipeline:
        1. Load pretrained NodeSTIDv2 backbone (frozen or finetunable).
        2. For each sample [B, L, N, C], run the backbone on each of the N
           nodes independently to get per-node hidden representations.
        3. Stack the N node representations into [B, hidden_dim, N, 1].
        4. Apply graph conditioning (GCN blocks with adaptive adjacency)
           to capture spatial dependencies.
        5. Regression head produces [B, output_len, N, 1].

    The backbone is the pretrained NodeSTIDv2 encoder (without the final
    regression layer). A new graph-conditioned head is trained on top.
    """

    def __init__(self, **model_args):
        super().__init__()

        # ---- Backbone config (NodeSTIDv2 params) ----
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]       # 3
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
        self.num_datasets = model_args.get("num_datasets", 4)
        self.dataset_id_dim = model_args.get("dataset_id_dim", 32)
        self.num_nodes = model_args["num_nodes"]
        self.freeze_backbone = model_args.get("freeze_backbone", True)
        # Default dataset ID to use when data doesn't have dataset_id channel
        # (e.g., when finetuning on a single dataset with 3 channels)
        self.default_dataset_id = model_args.get("default_dataset_id", 0)

        # ---- Build backbone (NodeSTIDv2 components) ----
        # Dataset ID embedding
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
        # Time series embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1), bias=True
        )
        # Backbone hidden dim
        self.backbone_hidden_dim = (
            self.embed_dim
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
            + self.dataset_id_dim * int(self.if_dataset_id)
        )
        # Backbone encoder
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.backbone_hidden_dim, self.backbone_hidden_dim)
              for _ in range(self.num_layer)]
        )

        # ---- Graph conditioning ----
        self.graph_hidden_dim = model_args.get("graph_hidden_dim", self.backbone_hidden_dim)
        self.num_graph_layers = model_args.get("num_graph_layers", 2)
        self.gcn_order = model_args.get("gcn_order", 2)
        self.gcn_dropout = model_args.get("gcn_dropout", 0.1)

        # Supports (adjacency matrices) passed at init
        supports = model_args.get("supports", None)
        self.supports = nn.ParameterList() if supports is None else None
        if supports is not None:
            # Register as buffers (not trainable)
            self.support_list = []
            for i, s in enumerate(supports):
                self.register_buffer(f'support_{i}', s if isinstance(s, torch.Tensor) else torch.tensor(s))
                self.support_list.append(f'support_{i}')
            self.num_supports = len(supports)
        else:
            self.num_supports = 0
            self.support_list = []

        # Adaptive adjacency (learned, like Graph WaveNet)
        self.use_adaptive_adj = model_args.get("use_adaptive_adj", True)
        if self.use_adaptive_adj:
            node_emb_dim = model_args.get("node_emb_dim", 10)
            self.nodevec1 = nn.Parameter(
                torch.randn(self.num_nodes, node_emb_dim), requires_grad=True)
            self.nodevec2 = nn.Parameter(
                torch.randn(node_emb_dim, self.num_nodes), requires_grad=True)
            total_supports = self.num_supports + 1
        else:
            total_supports = max(self.num_supports, 1)

        # Projection from backbone hidden to graph hidden if different
        if self.graph_hidden_dim != self.backbone_hidden_dim:
            self.proj = nn.Conv2d(self.backbone_hidden_dim, self.graph_hidden_dim,
                                  kernel_size=(1, 1), bias=True)
        else:
            self.proj = None

        # Graph conditioning blocks
        self.graph_blocks = nn.ModuleList([
            GraphConditioningBlock(
                self.graph_hidden_dim,
                support_len=total_supports,
                order=self.gcn_order,
                dropout=self.gcn_dropout
            ) for _ in range(self.num_graph_layers)
        ])

        # ---- Final regression head ----
        self.regression_layer = nn.Conv2d(
            in_channels=self.graph_hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1), bias=True
        )

    def load_pretrained_backbone(self, pretrained_path: str, strict: bool = False):
        """
        Load pretrained NodeSTIDv2 weights into the backbone layers.

        Args:
            pretrained_path: Path to the pretrained checkpoint (.pt/.pth file).
            strict: Whether to strictly enforce key matching.
        """
        pretrained_state = torch.load(pretrained_path, map_location='cpu')

        # Handle checkpoint wrappers (easytorch saves under 'model_state_dict')
        if 'model_state_dict' in pretrained_state:
            pretrained_state = pretrained_state['model_state_dict']

        # Map pretrained keys to this model's backbone keys
        backbone_keys = [
            'time_in_day_emb', 'day_in_week_emb', 'dataset_id_emb',
            'time_series_emb_layer', 'encoder'
        ]
        filtered_state = {}
        for k, v in pretrained_state.items():
            # Remove 'model.' prefix if present (from easytorch wrapper)
            clean_key = k.replace('model.', '') if k.startswith('model.') else k
            # Only load backbone parameters (skip regression_layer from pretrained)
            for bk in backbone_keys:
                if clean_key.startswith(bk):
                    filtered_state[clean_key] = v
                    break

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        print(f"[NodeSTIDGraphConditioned] Loaded pretrained backbone from {pretrained_path}")
        print(f"  Loaded keys: {len(filtered_state)}")
        if missing:
            # Filter out expected missing keys (graph layers, regression, etc.)
            truly_missing = [k for k in missing if any(k.startswith(bk) for bk in backbone_keys)]
            if truly_missing:
                print(f"  Missing backbone keys: {truly_missing}")

        # Optionally freeze backbone
        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        backbone_params = [
            self.time_series_emb_layer,
            self.encoder,
        ]
        if self.if_time_in_day:
            self.time_in_day_emb.requires_grad_(False)
        if self.if_day_in_week:
            self.day_in_week_emb.requires_grad_(False)
        if self.if_dataset_id:
            self.dataset_id_emb.requires_grad_(False)
        for module in backbone_params:
            for param in module.parameters():
                param.requires_grad_(False)
        print("[NodeSTIDGraphConditioned] Backbone frozen.")

    def _get_supports(self):
        """Build the list of support matrices including adaptive adj."""
        supports = []
        for name in self.support_list:
            supports.append(getattr(self, name))
        if self.use_adaptive_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports.append(adp)
        return supports

    def _backbone_forward_per_node(self, history_data: torch.Tensor) -> torch.Tensor:
        """
        Run the pretrained backbone on all N nodes in parallel.

        Input: history_data [B, L, N, C] where C >= 4
        Output: hidden [B, backbone_hidden_dim, N, 1]

        The backbone treats each node independently — we process all N
        nodes at once by reshaping.
        """
        B, L, N, C = history_data.shape

        # ---- Extract per-node time series features ----
        input_data = history_data[..., :self.input_dim]  # [B, L, N, 3]

        # ---- Temporal embeddings (shared across nodes at same time step) ----
        if self.if_time_in_day:
            t_i_d = history_data[..., 1]  # [B, L, N]
            tid_idx = (t_i_d[:, -1, :] * self.time_of_day_size).long()  # [B, N]
            time_in_day_emb = self.time_in_day_emb[tid_idx]  # [B, N, tid_dim]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w = history_data[..., 2]  # [B, L, N]
            diw_idx = (d_i_w[:, -1, :] * self.day_of_week_size).long()  # [B, N]
            day_in_week_emb = self.day_in_week_emb[diw_idx]  # [B, N, diw_dim]
        else:
            day_in_week_emb = None

        if self.if_dataset_id:
            if C > 3:
                # Dataset ID is provided as channel 3
                ds_ids = history_data[:, -1, :, 3].long()  # [B, N]
            else:
                # No dataset_id channel — use default
                ds_ids = torch.full((B, N), self.default_dataset_id,
                                    dtype=torch.long, device=history_data.device)
            dataset_emb = self.dataset_id_emb[ds_ids]  # [B, N, ds_dim]
        else:
            dataset_emb = None

        # ---- Time series embedding ----
        # [B, L, N, 3] -> [B, N, L, 3] -> [B, N, L*3] -> [B, L*3, N, 1]
        x = input_data.transpose(1, 2).contiguous()          # [B, N, L, 3]
        x = x.view(B, N, -1).transpose(1, 2).unsqueeze(-1)   # [B, L*3, N, 1]
        time_series_emb = self.time_series_emb_layer(x)       # [B, embed_dim, N, 1]

        # ---- Collect embeddings ----
        embeddings = [time_series_emb]
        if time_in_day_emb is not None:
            embeddings.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))  # [B, tid, N, 1]
        if day_in_week_emb is not None:
            embeddings.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))  # [B, diw, N, 1]
        if dataset_emb is not None:
            embeddings.append(dataset_emb.transpose(1, 2).unsqueeze(-1))      # [B, ds, N, 1]

        hidden = torch.cat(embeddings, dim=1)  # [B, backbone_hidden_dim, N, 1]

        # ---- Backbone encoder (node-independent MLPs via Conv2d 1x1) ----
        hidden = self.encoder(hidden)  # [B, backbone_hidden_dim, N, 1]

        return hidden

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor,
                batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Forward pass of NodeSTIDGraphConditioned.

        Args:
            history_data: [B, L, N, C] standard spatial-temporal input
        Returns:
            prediction: [B, output_len, N, 1]
        """

        # Step 1: Backbone produces per-node hidden representations
        hidden = self._backbone_forward_per_node(history_data)  # [B, backbone_hidden_dim, N, 1]

        # Step 2: Project to graph hidden dim if needed
        if self.proj is not None:
            hidden = self.proj(hidden)  # [B, graph_hidden_dim, N, 1]

        # Step 3: Graph conditioning
        supports = self._get_supports()
        for block in self.graph_blocks:
            hidden = block(hidden, supports)  # [B, graph_hidden_dim, N, 1]

        # Step 4: Regression
        prediction = self.regression_layer(hidden)  # [B, output_len, N, 1]

        return prediction

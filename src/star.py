import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .multi_attention_forward import multi_head_attention_forward


def get_noise(shape, noise_type, device=None):
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if noise_type == "gaussian":
        return torch.randn(shape, device=device)
    elif noise_type == "uniform":
        return torch.rand(*shape, device=device).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        atts = []

        for i in range(self.num_layers):
            output, attn = self.layers[i](output, src_mask=mask,
                                          src_key_padding_mask=src_key_padding_mask)
            atts.append(attn)
        if self.norm:
            output = self.norm(output)

        return output


class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def forward(self, src, mask):
        n_mask = mask + torch.eye(mask.shape[0], mask.shape[0], device=mask.device)
        n_mask = n_mask.float().masked_fill(n_mask == 0., float(-1e20)).masked_fill(n_mask == 1., float(0.0))
        output = self.transformer_encoder(src, mask=n_mask)

        return output


class PositionalEncoding(nn.Module):
    """Sinusoidal time-position encoding for tensors shaped [T, N, D]."""

    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x):
        """Add the first T positional vectors to x.

        Args:
            x: Temporal features with shape [T, N, D].

        Returns:
            Tensor with the same shape [T, N, D].
        """
        return x + self.pe[:x.size(0)].type_as(x)


# ---------------------------------------------------------------------------
# STAR-ART-ViTE v2 modules.
#
# Shape convention used below:
#   T  = observed length, normally 8
#   P  = prediction length, normally 12
#   N  = all agents in the batch
#   Nv = valid observed agents after masking by seq_list
#   D  = model hidden dimension, normally 64
#   H  = number of attention heads
#   M  = number of virtual nodes
#   K  = number of trajectory decoder heads / candidate futures
# ---------------------------------------------------------------------------


class TemporalAwareRelationGraph(nn.Module):
    """ART-style temporal-aware pairwise relation graph.

    This module builds a dense relation graph inside each scene slice. It
    compares two agents across the observed time axis, so the edge from i to j
    is based on the full observed motion pattern instead of only one distance
    value at the last frame.

    Inputs:
        temporal_features: [T, Nv, D] temporal encoder output.
        scene_slices: List of (start, end) ranges in compact valid-agent space.

    Outputs:
        weights: [Nv, Nv] learned relation strength W_ij, zero across scenes.
        relations: [Nv, Nv, D] edge feature R_ij for edge-aware attention.
    """

    def __init__(self, model_dim, num_heads):
        super(TemporalAwareRelationGraph, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(model_dim, model_dim * 3)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.edge_score = nn.Linear(model_dim * 2, 1)

    def forward(self, temporal_features, scene_slices):
        """Compute pairwise temporal attention and edge scores per scene."""
        # temporal_features: [T, Nv, D].
        time_steps, num_nodes, model_dim = temporal_features.shape
        device = temporal_features.device

        # Allocate full compact-agent tensors. Cross-scene blocks stay zero.
        # relations: [Nv, Nv, D], one D-dim edge vector for every ordered pair.
        # weights: [Nv, Nv], one scalar relation strength for every ordered pair.
        relations = temporal_features.new_zeros(num_nodes, num_nodes, model_dim)
        weights = temporal_features.new_zeros(num_nodes, num_nodes)

        if num_nodes == 0:
            return weights, relations

        # qkv: [T, Nv, 3D].
        qkv = self.qkv(temporal_features)
        # q, k, v: each [T, Nv, D].
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 1: view [T, Nv, D] -> [T, Nv, H, Dh].
        # Step 2: permute [T, Nv, H, Dh] -> [H, Nv, T, Dh].
        # After this, q[h, i, t, d] means head h, agent i, time t, feature d.
        q = q.view(time_steps, num_nodes, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        k = k.view(time_steps, num_nodes, self.num_heads, self.head_dim).permute(2, 1, 0, 3)
        v = v.view(time_steps, num_nodes, self.num_heads, self.head_dim).permute(2, 1, 0, 3)

        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 1:
                continue

            # Slice one scene block:
            # q_scene, k_scene, v_scene: [H, Ns, T, Dh].
            q_scene = q[:, st:ed]
            k_scene = k[:, st:ed]
            v_scene = v[:, st:ed]

            # einsum "hitd,hjtd->hijt":
            #   q_scene index letters: h i t d = [H, target_i, T, Dh]
            #   k_scene index letters: h j t d = [H, source_j, T, Dh]
            #   output letters:        h i j t = [H, target_i, source_j, T]
            # The letter d disappears from output, so einsum sums over d:
            #   scores[h, i, j, t] = sum_d q[h, i, t, d] * k[h, j, t, d].
            scores = torch.einsum("hitd,hjtd->hijt", q_scene, k_scene) * self.scale
            # time_alpha: [H, Ns, Ns, T]. Softmax is along T, so for each
            # fixed (h, i, j), time_alpha[h, i, j, :].sum() == 1.
            time_alpha = F.softmax(scores, dim=-1)

            # einsum "hijt,hjtd->hijd":
            #   time_alpha letters: h i j t = [H, target_i, source_j, T]
            #   v_scene letters:    h j t d = [H, source_j, T, Dh]
            #   output letters:     h i j d = [H, target_i, source_j, Dh]
            # The letter t disappears from output, so einsum sums over time:
            #   relation[h, i, j, d] =
            #       sum_t time_alpha[h, i, j, t] * v[h, j, t, d].
            relation = torch.einsum("hijt,hjtd->hijd", time_alpha, v_scene)

            # relation before permute: [H, Ns, Ns, Dh].
            # permute(1, 2, 0, 3): [H, Ns, Ns, Dh] -> [Ns, Ns, H, Dh].
            # view(Ns, Ns, D): merge H * Dh back into D, so [Ns, Ns, H, Dh] -> [Ns, Ns, D].
            relation = relation.permute(1, 2, 0, 3).contiguous().view(scene_nodes, scene_nodes, model_dim)
            # out_proj keeps the same edge shape: [Ns, Ns, D] -> [Ns, Ns, D].
            relation = self.out_proj(relation)

            # Score an edge using both directions, so W_ij can depend on how
            # i attends to j and how j attends back to i.
            # relation[i, j, :] is the feature for source j -> target i.
            # reversed_relation[i, j, :] equals relation[j, i, :], the opposite direction.
            reversed_relation = relation.transpose(0, 1)
            # cat on dim=-1: [Ns, Ns, D] + [Ns, Ns, D] -> [Ns, Ns, 2D].
            # edge_score: [Ns, Ns, 2D] -> [Ns, Ns, 1].
            # squeeze(-1): [Ns, Ns, 1] -> [Ns, Ns].
            edge_logits = self.edge_score(torch.cat([relation, reversed_relation], dim=-1)).squeeze(-1)
            # edge_weight: [Ns, Ns], values in (0, 1).
            edge_weight = torch.sigmoid(edge_logits)

            # The interaction graph is directed but self edges are handled
            # separately by later attention layers, so remove them here.
            eye = torch.eye(scene_nodes, dtype=torch.bool, device=device)
            edge_weight = edge_weight.masked_fill(eye, 0.0)

            # Write this scene block back into the full compact-agent tensors:
            # relations[st:ed, st:ed]: [Ns, Ns, D]
            # weights[st:ed, st:ed]: [Ns, Ns]
            relations[st:ed, st:ed] = relation
            weights[st:ed, st:ed] = edge_weight

        return weights, relations


class AdaptiveInteractionPruning(nn.Module):
    """Top-p pruning over learned interaction strengths.

    Given a dense relation matrix, this module keeps the smallest high-weight
    neighbor set whose cumulative edge mass crosses top_p for each target row.
    It returns a row-normalized sparse adjacency matrix. Rows with no positive
    edges remain all zero.

    Inputs:
        weights: [Nv, Nv] relation strength after ART/spatial mixing.
        scene_slices: Scene boundaries in compact valid-agent space.

    Outputs:
        pruned: [Nv, Nv] sparse row-normalized adjacency.
        density: Scalar ratio of kept edges among all possible within-scene
            non-self edges.
    """

    def __init__(self, top_p=0.75, eps=1e-8):
        super(AdaptiveInteractionPruning, self).__init__()
        self.top_p = float(top_p)
        self.eps = eps

    def forward(self, weights, scene_slices):
        """Apply per-row top-p cumulative pruning inside each scene."""
        # weights: [Nv, Nv]. pruned keeps the same full compact-agent shape.
        pruned = torch.zeros_like(weights)

        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 1:
                continue

            # scene_weights: [Ns, Ns], one within-scene relation matrix.
            scene_weights = weights[st:ed, st:ed].clone()
            eye = torch.eye(scene_nodes, dtype=torch.bool, device=weights.device)
            scene_weights = scene_weights.masked_fill(eye, 0.0)

            # Only rows with positive relation mass can be normalized.
            # row_totals: [Ns, 1]. active_rows: [Ns].
            row_totals = scene_weights.sum(dim=-1, keepdim=True)
            active_rows = row_totals.squeeze(-1) > self.eps
            if not bool(active_rows.any()):
                continue

            # Sort each target row from strongest neighbor to weakest neighbor.
            # sorted_values: [Ns, Ns], sorted_indices: [Ns, Ns].
            sorted_values, sorted_indices = torch.sort(scene_weights, descending=True, dim=-1)
            # cumulative: [Ns, Ns]. thresholds: [Ns, 1].
            cumulative = torch.cumsum(sorted_values, dim=-1)
            thresholds = row_totals * self.top_p
            # keep_sorted: [Ns, Ns], mask in sorted-neighbor order.
            keep_sorted = cumulative <= thresholds

            # Also keep the first edge that crosses the threshold, preventing
            # active rows from becoming empty when the top edge already exceeds
            # top_p of the row mass.
            over_threshold = cumulative > thresholds
            fallback = torch.full((scene_nodes, 1), scene_nodes - 1, dtype=torch.long, device=weights.device)
            # first_over: [Ns, 1], one selected sorted-column index per target row.
            first_over = torch.where(
                over_threshold.any(dim=-1, keepdim=True),
                over_threshold.float().argmax(dim=-1, keepdim=True),
                fallback,
            )
            # scatter_ marks the first over-threshold edge as kept in each row.
            keep_sorted.scatter_(1, first_over, True)
            # active_rows.unsqueeze(-1): [Ns] -> [Ns, 1], broadcast to [Ns, Ns].
            keep_sorted = keep_sorted & active_rows.unsqueeze(-1)

            # Scatter the sorted mask back to original neighbor order and
            # normalize every active row to sum to 1.
            # selected starts in original neighbor order: [Ns, Ns].
            selected = torch.zeros_like(scene_weights)
            # scatter along dim=1 uses sorted_indices to put kept weights back
            # at their original neighbor columns.
            selected.scatter_(1, sorted_indices, sorted_values * keep_sorted.type_as(sorted_values))
            selected = selected / selected.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            pruned[st:ed, st:ed] = selected

        possible_edges = sum(max(ed - st, 0) * max(ed - st - 1, 0) for st, ed in scene_slices)
        if possible_edges > 0:
            density = pruned.gt(0).float().sum() / float(possible_edges)
        else:
            density = pruned.new_tensor(0.0)

        return pruned, density


class RelationalTransformerLayer(nn.Module):
    """Edge-aware attention over the adaptively pruned relation graph.

    This is a native-PyTorch graph attention layer. It uses node features as
    Q/K/V, injects ART edge features into K/V, and uses the sparse AIP
    adjacency as both an attention mask and a log-probability bias.

    Inputs:
        node_features: [Nv, D] node features.
        edge_features: [Nv, Nv, D] pairwise relation features.
        adjacency: [Nv, Nv] row-normalized sparse edge weights.

    Returns:
        Updated node features [Nv, D].
    """

    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(RelationalTransformerLayer, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.edge_k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.edge_v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
        )

    def forward(self, node_features, edge_features, adjacency):
        """Run one edge-aware self-attention block plus residual FFN."""
        # node_features: [Nv, D].
        # edge_features: [Nv, Nv, D].
        # adjacency: [Nv, Nv].
        num_nodes = node_features.size(0)
        if num_nodes == 0:
            return node_features

        # Linear projections keep [Nv, D].
        # view: [Nv, D] -> [Nv, H, Dh].
        # permute: [Nv, H, Dh] -> [H, Nv, Dh].
        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim).permute(1, 0, 2)

        # Edge projections keep [Nv, Nv, D].
        # view: [Nv, Nv, D] -> [Nv, Nv, H, Dh].
        edge_k = self.edge_k_proj(edge_features).view(num_nodes, num_nodes, self.num_heads, self.head_dim)
        edge_v = self.edge_v_proj(edge_features).view(num_nodes, num_nodes, self.num_heads, self.head_dim)
        # permute: [Nv, Nv, H, Dh] -> [H, Nv, Nv, Dh].
        edge_k = edge_k.permute(2, 0, 1, 3)
        edge_v = edge_v.permute(2, 0, 1, 3)

        # scores[h, i, j] compares target node i with source node j plus edge R_ij.
        # q[:, :, None, :]: [H, Nv, 1, Dh], broadcasts target i over all source j.
        # k[:, None, :, :]: [H, 1, Nv, Dh], broadcasts source j over all target i.
        # edge_k: [H, Nv, Nv, Dh].
        # product result: [H, Nv, Nv, Dh], then sum over Dh -> [H, Nv, Nv].
        scores = (q[:, :, None, :] * (k[:, None, :, :] + edge_k)).sum(dim=-1) * self.scale
        eye = torch.eye(num_nodes, dtype=torch.bool, device=node_features.device)

        # Self edges are always visible; non-self edges must survive AIP pruning.
        # attention_mask: [Nv, Nv], broadcast to [H, Nv, Nv] below.
        attention_mask = adjacency.gt(0) | eye
        scores = scores.masked_fill(~attention_mask.unsqueeze(0), -1e20)

        # AIP weights become log-biases: stronger retained edges get larger logits.
        # edge_bias: [Nv, Nv], broadcast over H when added to scores.
        edge_bias = torch.where(adjacency.gt(0), torch.log(adjacency.clamp_min(1e-8)), torch.zeros_like(adjacency))
        scores = scores + edge_bias.unsqueeze(0)

        # attention: [H, Nv, Nv], softmax over source j for each target i.
        attention = F.softmax(scores, dim=-1)
        # v[:, None, :, :]: [H, 1, Nv, Dh], broadcast to [H, Nv, Nv, Dh].
        # values: [H, Nv, Nv, Dh], source value plus edge value.
        values = v[:, None, :, :] + edge_v

        # einsum "hij,hijd->hid":
        #   attention: h i j = [H, target_i, source_j]
        #   values:    h i j d = [H, target_i, source_j, Dh]
        #   output:    h i d = [H, target_i, Dh]
        # The letter j disappears, so values are summed over source agents.
        attention_out = torch.einsum("hij,hijd->hid", attention, values)
        # permute: [H, Nv, Dh] -> [Nv, H, Dh].
        # view: [Nv, H, Dh] -> [Nv, D].
        attention_out = attention_out.permute(1, 0, 2).contiguous().view(num_nodes, self.model_dim)
        # out_proj keeps [Nv, D].
        attention_out = self.out_proj(attention_out)

        # Standard Transformer residual structure.
        node_features = self.norm1(node_features + self.dropout(attention_out))
        node_features = self.norm2(node_features + self.dropout(self.ffn(node_features)))
        return node_features


class OneHopExpert(nn.Module):
    """First-order expert that aggregates messages on the pruned graph.

    This expert models direct one-hop interactions. AIP provides the candidate
    sparse graph, and this module learns an additional edge gate so direct
    neighbors can be reweighted based on node states.

    Inputs:
        node_features: [Nv, D].
        adjacency: [Nv, Nv] from AdaptiveInteractionPruning.

    Returns:
        One-hop expert output [Nv, D].
    """

    def __init__(self, model_dim, dropout=0.1):
        super(OneHopExpert, self).__init__()
        self.message = nn.Linear(model_dim, model_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, node_features, adjacency):
        """Aggregate gated one-hop messages and update each node."""
        # node_features: [Nv, D].
        # adjacency: [Nv, Nv].
        num_nodes = node_features.size(0)
        if num_nodes == 0:
            return node_features

        # node_features[:, None, :]: [Nv, 1, D] -> expand to [Nv, Nv, D].
        # node_features[None, :, :]: [1, Nv, D] -> expand to [Nv, Nv, D].
        # target[i, j, :] is h_i; source[i, j, :] is h_j.
        target = node_features[:, None, :].expand(num_nodes, num_nodes, -1)
        source = node_features[None, :, :].expand(num_nodes, num_nodes, -1)
        # cat: [Nv, Nv, D] + [Nv, Nv, D] -> [Nv, Nv, 2D].
        # edge_gate: [Nv, Nv, 2D] -> [Nv, Nv, 1], squeeze -> [Nv, Nv].
        learned_gate = self.edge_gate(torch.cat([target, source], dim=-1)).squeeze(-1)

        # Preserve only AIP-selected edges, then renormalize row-wise.
        # effective_adjacency: [Nv, Nv], row-normalized over source j.
        effective_adjacency = adjacency * learned_gate
        effective_adjacency = effective_adjacency / effective_adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # message(node_features): [Nv, D].
        # matmul: [Nv, Nv] @ [Nv, D] -> [Nv, D].
        message = torch.matmul(effective_adjacency, self.message(node_features))
        # cat: node [Nv, D] + message [Nv, D] -> [Nv, 2D].
        # update: [Nv, 2D] -> [Nv, D].
        update = self.update(torch.cat([node_features, message], dim=-1))
        return self.norm(node_features + update)


class HighOrderExpert(nn.Module):
    """Virtual-node expert for efficient high-order interactions.

    This expert follows the ViTE idea: a few learnable virtual nodes first read
    from all real nodes in a scene, then real nodes read back from the virtual
    nodes. This gives each real node access to global/high-order context without
    stacking many graph layers.

    Inputs:
        node_features: [Nv, D].
        scene_slices: Scene boundaries, so virtual-node exchange never crosses
            unrelated scenes in a merged batch.

    Returns:
        High-order expert output [Nv, D].
    """

    def __init__(self, model_dim, num_virtual_nodes=3, dropout=0.1):
        super(HighOrderExpert, self).__init__()
        self.model_dim = model_dim
        self.num_virtual_nodes = num_virtual_nodes
        self.scale = model_dim ** -0.5

        self.virtual_nodes = nn.Parameter(torch.randn(num_virtual_nodes, model_dim) * 0.02)
        self.virtual_query = nn.Linear(model_dim, model_dim)
        self.real_key = nn.Linear(model_dim, model_dim)
        self.real_value = nn.Linear(model_dim, model_dim)
        self.virtual_update = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.real_query = nn.Linear(model_dim, model_dim)
        self.virtual_key = nn.Linear(model_dim, model_dim)
        self.virtual_value = nn.Linear(model_dim, model_dim)
        self.real_update = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.norm_virtual = nn.LayerNorm(model_dim)
        self.norm_real = nn.LayerNorm(model_dim)

    def forward(self, node_features, scene_slices):
        """Run real-to-virtual and virtual-to-real attention per scene."""
        # node_features: [Nv, D]. output keeps [Nv, D].
        output = torch.zeros_like(node_features)

        for st, ed in scene_slices:
            # real_nodes: [Ns, D] for one scene.
            real_nodes = node_features[st:ed]
            if real_nodes.size(0) == 0:
                continue

            # virtual_nodes: [M, D], shared learnable tokens copied to this dtype/device.
            virtual_nodes = self.virtual_nodes.type_as(real_nodes)

            # Real-to-virtual: [M, D] attends over [Ns, D], giving virtual
            # summaries of the whole scene.
            # virtual_query(virtual_nodes): [M, D].
            # real_key(real_nodes).transpose(0, 1): [D, Ns].
            # matmul -> virtual_attention logits [M, Ns].
            virtual_attention = torch.matmul(
                self.virtual_query(virtual_nodes),
                self.real_key(real_nodes).transpose(0, 1),
            ) * self.scale
            # softmax over real nodes: every virtual node reads a distribution over Ns agents.
            virtual_attention = F.softmax(virtual_attention, dim=-1)
            # real_value(real_nodes): [Ns, D].
            # matmul: [M, Ns] @ [Ns, D] -> [M, D].
            virtual_context = torch.matmul(virtual_attention, self.real_value(real_nodes))
            # virtual_update keeps [M, D]; virtual_nodes stays [M, D].
            virtual_nodes = self.norm_virtual(virtual_nodes + self.virtual_update(virtual_context))

            # Virtual-to-real: every real node reads compact high-order context
            # from the updated M virtual nodes.
            # real_query(real_nodes): [Ns, D].
            # virtual_key(virtual_nodes).transpose(0, 1): [D, M].
            # matmul -> real_attention logits [Ns, M].
            real_attention = torch.matmul(
                self.real_query(real_nodes),
                self.virtual_key(virtual_nodes).transpose(0, 1),
            ) * self.scale
            # softmax over virtual nodes: each real node chooses among M global tokens.
            real_attention = F.softmax(real_attention, dim=-1)
            # virtual_value(virtual_nodes): [M, D].
            # matmul: [Ns, M] @ [M, D] -> [Ns, D].
            real_context = torch.matmul(real_attention, self.virtual_value(virtual_nodes))
            # real_update keeps [Ns, D], written back to output[st:ed].
            output[st:ed] = self.norm_real(real_nodes + self.real_update(real_context))

        return output


class ExpertRouter(nn.Module):
    """Noisy top-p router over one-hop and high-order experts.

    The router chooses between two experts for each node:
        expert 0: OneHopExpert
        expert 1: HighOrderExpert

    In training, Gaussian noise encourages exploration. Top-p selection can
    keep one expert when the router is confident, or both experts when their
    probabilities are close.

    Inputs:
        node_features: [Nv, D].

    Outputs:
        weights: [Nv, 2] normalized selected expert weights.
        router_loss: scalar CV-squared balance loss over expert importance.
        expert_usage: [2] fraction of nodes selecting each expert.
    """

    def __init__(self, model_dim, top_p=0.7, noise_epsilon=1e-2):
        super(ExpertRouter, self).__init__()
        self.top_p = float(top_p)
        self.noise_epsilon = float(noise_epsilon)
        self.gate = nn.Linear(model_dim, 2, bias=False)
        self.noise = nn.Linear(model_dim, 2, bias=False)
        self.softplus = nn.Softplus()

    @staticmethod
    def cv_squared(x):
        """Coefficient-of-variation squared for load-balance regularization."""
        eps = 1e-10
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        return x.float().var(unbiased=False) / (x.float().mean() ** 2 + eps)

    def forward(self, node_features):
        """Compute noisy top-p expert weights for each node."""
        # node_features: [Nv, D].
        if node_features.size(0) == 0:
            empty = node_features.new_zeros(0, 2)
            return empty, node_features.new_tensor(0.0), node_features.new_zeros(2)

        # logits: [Nv, 2], one score for one-hop expert and one for high-order expert.
        logits = self.gate(node_features)
        if self.training:
            # Noisy gating follows MoE practice: uncertainty helps prevent
            # early collapse to one expert during training.
            # noise_std: [Nv, 2].
            noise_std = self.softplus(self.noise(node_features)) + self.noise_epsilon
            logits = logits + torch.randn_like(logits) * noise_std

        # probs: [Nv, 2], each row sums to 1 over two experts.
        probs = F.softmax(logits, dim=-1)

        # Sort two expert probabilities and keep the smallest prefix whose
        # cumulative mass crosses top_p. With two experts this means either
        # single-expert routing or two-expert mixing.
        # sorted_probs/sorted_indices: [Nv, 2].
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        # cumulative: [Nv, 2].
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # keep_sorted: [Nv, 2] in sorted-expert order.
        keep_sorted = cumulative <= self.top_p
        over_threshold = cumulative > self.top_p
        # first_over: [Nv, 1], selected sorted expert index that crosses top_p.
        first_over = over_threshold.float().argmax(dim=-1, keepdim=True)
        keep_sorted.scatter_(1, first_over, True)

        # selected: [Nv, 2] in original expert order.
        selected = torch.zeros_like(probs)
        selected.scatter_(1, sorted_indices, keep_sorted.type_as(probs))
        # weights: [Nv, 2], zero for unselected experts and row-normalized over selected ones.
        weights = probs * selected
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Importance uses soft probabilities rather than hard selected weights,
        # producing a smooth load-balance loss.
        importance = probs.sum(dim=0)
        router_loss = self.cv_squared(importance)
        expert_usage = selected.gt(0).float().mean(dim=0)
        return weights, router_loss, expert_usage


class TrajectoryDecoder(nn.Module):
    """One residual trajectory decoder head.

    Each decoder predicts residual offsets around a constant-velocity baseline.
    Multiple instances of this module form K candidate future heads.

    Inputs to forward:
        features: [Nv, 3D] concatenated node features.
        current_pos: [Nv, 2] last observed normalized position.
        current_velocity: [Nv, 2] last observed velocity.

    Returns:
        trajectories: [P, Nv, 2] one candidate future.
    """

    def __init__(self, input_dim, hidden_dim, pred_length, dropout=0.1):
        super(TrajectoryDecoder, self).__init__()
        self.pred_length = pred_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pred_length * 2),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, current_pos, current_velocity, use_motion_prior=True, motion_gate=None):
        """Decode one candidate trajectory.

        If use_motion_prior=True, the decoder predicts residuals around the
        constant-velocity baseline. If False, the decoder predicts residuals
        around the last observed position only, which is useful for the
        no_motion_prior ablation. motion_gate can softly scale the velocity
        baseline per agent for adaptive-prior experiments.
        """
        # features: [Nv, 3D].
        # net(features): [Nv, P * 2].
        # view: [Nv, P * 2] -> [Nv, P, 2].
        # Residual is small at initialization because the final layer is
        # initialized near zero; early training starts from motion prior.
        residual = self.net(features).view(features.size(0), self.pred_length, 2)
        # steps: [P] -> [1, P, 1], broadcasts over Nv and xy coordinates.
        steps = torch.arange(1, self.pred_length + 1, device=features.device, dtype=features.dtype)
        steps = steps.view(1, self.pred_length, 1)

        if use_motion_prior:
            if motion_gate is not None:
                # motion_gate: [Nv, 1] in [0, 1].
                # [Nv, 2] * [Nv, 1] -> [Nv, 2].
                current_velocity = current_velocity * motion_gate
            # baseline: [Nv, P, 2] = last position + step index * last velocity.
            # current_pos.unsqueeze(1): [Nv, 2] -> [Nv, 1, 2].
            # current_velocity.unsqueeze(1): [Nv, 2] -> [Nv, 1, 2].
            # steps * velocity broadcasts to [Nv, P, 2].
            baseline = current_pos.unsqueeze(1) + steps * current_velocity.unsqueeze(1)
        else:
            # baseline: [Nv, 1, 2] -> [Nv, P, 2]. This removes velocity
            # extrapolation but still predicts normalized futures relative to
            # the last observed position.
            baseline = current_pos.unsqueeze(1).expand(-1, self.pred_length, -1)
        # trajectories: [Nv, P, 2].
        trajectories = baseline + residual

        # Return [P, Nv, 2] so stacking K heads gives [K, P, Nv, 2].
        return trajectories.permute(1, 0, 2).contiguous()


class STAR(torch.nn.Module):
    """STAR-ART-ViTE v2 model.

    This class keeps the STAR dataloader interface but replaces the original
    model body with:
        1. observed position/velocity temporal encoding,
        2. ART-style temporal-aware relation graph,
        3. STAR-style spatial prior mixing,
        4. adaptive top-p interaction pruning,
        5. edge-aware relational Transformer layers,
        6. ViTE-style one-hop/high-order expert routing,
        7. optional adaptive motion/spatial gates,
        8. K residual trajectory decoder heads.

    forward(inputs) returns:
        pred: [K, P, N, 2] normalized-coordinate candidate futures.
        aux: dict with router_loss, aip_density, expert_usage.
    """

    def __init__(self, args, dropout_prob=0):
        super(STAR, self).__init__()
        self.args = args
        self.obs_length = int(getattr(args, "obs_length", 8))
        self.pred_length = int(getattr(args, "pred_length", 12))
        self.sample_num = int(getattr(args, "sample_num", 20))
        self.model_dim = int(getattr(args, "model_dim", 64))
        self.num_heads = int(getattr(args, "num_heads", 4))
        self.rt_layers = int(getattr(args, "rt_layers", 1))
        self.num_virtual_nodes = int(getattr(args, "num_virtual_nodes", 3))
        self.dropout_prob = float(dropout_prob if dropout_prob else getattr(args, "dropout", 0.1))
        self.spatial_prior_mix = float(getattr(args, "spatial_prior_mix", 0.6))
        self.spatial_sigma = float(getattr(args, "spatial_sigma", 2.0))
        self.ablation = str(getattr(args, "ablation", "full")).lower()
        valid_ablations = {
            "full",
            "motion_only",
            "spatial_only",
            "art_only",
            "art_aip",
            "vite_only",
            "no_motion_prior",
            "adaptive",
            "adaptive_no_motion",
            "adaptive_no_spatial",
            "adaptive_motion_only",
        }
        if self.ablation not in valid_ablations:
            raise ValueError("Unknown ablation mode: {}".format(self.ablation))
        self.motion_mode = "constant"
        if self.ablation == "no_motion_prior":
            self.motion_mode = "none"
        elif self.ablation in ("adaptive", "adaptive_no_spatial", "adaptive_motion_only"):
            self.motion_mode = "gated"
        self.use_motion_prior = self.motion_mode != "none"

        # Per-frame input is [x, y, vx, vy] -> D.
        self.input_embedding = nn.Linear(4, self.model_dim)
        self.position_encoding = PositionalEncoding(self.model_dim, max_len=max(self.obs_length + 1, 32))

        # Temporal encoder consumes [T, Nv, D] and keeps the same shape.
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.model_dim * 4,
            dropout=self.dropout_prob,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(self.model_dim)

        # Graph construction and edge-aware interaction modules.
        self.relation_graph = TemporalAwareRelationGraph(self.model_dim, self.num_heads)
        self.interaction_pruning = AdaptiveInteractionPruning(getattr(args, "aip_top_p", 0.75))
        self.relational_layers = nn.ModuleList([
            RelationalTransformerLayer(self.model_dim, self.num_heads, self.dropout_prob)
            for _ in range(self.rt_layers)
        ])

        # Two-expert MoE router: expert 0 is local one-hop, expert 1 is virtual high-order.
        self.router = ExpertRouter(
            self.model_dim,
            top_p=getattr(args, "router_top_p", 0.7),
            noise_epsilon=getattr(args, "noise_epsilon", 1e-2),
        )
        self.one_hop_expert = OneHopExpert(self.model_dim, self.dropout_prob)
        self.high_order_expert = HighOrderExpert(
            self.model_dim,
            num_virtual_nodes=self.num_virtual_nodes,
            dropout=self.dropout_prob,
        )
        self.router_norm = nn.LayerNorm(self.model_dim)

        # Learnable scale for constant-velocity motion prior.
        self.velocity_scale = nn.Parameter(torch.ones(1))

        # New adaptive-prior gates. Spatial gate is used before ViTE one-hop
        # aggregation; motion gate is used inside the decoder baseline.
        self.motion_context_dim = 4
        self.spatial_gate = nn.Sequential(
            nn.Linear(self.model_dim + self.motion_context_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.model_dim, 1),
            nn.Sigmoid(),
        )
        self.motion_gate = nn.Sequential(
            nn.Linear(self.model_dim * 3 + self.motion_context_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.model_dim, 1),
            nn.Sigmoid(),
        )
        # Start close to vite_only: prefer constant velocity and weakly mix
        # spatial adjacency at initialization, then let data move the gates.
        nn.init.constant_(self.motion_gate[-2].bias, float(getattr(args, "motion_gate_bias", 1.0)))
        nn.init.constant_(self.spatial_gate[-2].bias, float(getattr(args, "spatial_gate_bias", -1.0)))

        # Decoder input concatenates initial, relational, and routed features: [Nv, 3D].
        decoder_input_dim = self.model_dim * 3
        decoder_hidden_dim = int(getattr(args, "decoder_hidden_dim", self.model_dim * 2))
        self.decoder_heads = nn.ModuleList([
            TrajectoryDecoder(decoder_input_dim, decoder_hidden_dim, self.pred_length, self.dropout_prob)
            for _ in range(self.sample_num)
        ])

    def _valid_observed_mask(self, seq_list):
        """Return agents present throughout all observed frames.

        Args:
            seq_list: [T, N] binary existence mask from the dataloader.

        Returns:
            valid_mask: [N] bool mask. Only valid agents are encoded by the
                model; invalid agents keep zero predictions and are masked in loss.
        """
        # obs_seq: [T_obs, N].
        obs_seq = seq_list[:self.obs_length]
        # gt(0): [T_obs, N] bool. all(dim=0): [N] bool.
        return obs_seq.gt(0).all(dim=0)

    def _scene_slices(self, batch_pednum, valid_mask):
        """Map original batch scenes to compact valid-agent scene slices.

        STAR batches can merge several independent scenes along the agent
        dimension. After removing invalid agents, this function rebuilds scene
        boundaries in compact [Nv] space so graph edges never cross scenes.

        Args:
            batch_pednum: [num_scenes] number of original agents per scene.
            valid_mask: [N] bool mask over original agents.

        Returns:
            List[(start, end)] where 0 <= start < end <= Nv.
        """
        counts = batch_pednum.detach().cpu().long().tolist()
        scene_slices = []
        original_start = 0
        compact_start = 0

        for count in counts:
            # original_start/original_end index the original N-agent axis.
            original_end = original_start + int(count)
            # kept is how many agents survive valid_mask inside this scene.
            kept = int(valid_mask[original_start:original_end].sum().item())
            if kept > 0:
                # compact_start indexes the compact Nv-agent axis after masking.
                scene_slices.append((compact_start, compact_start + kept))
            compact_start += kept
            original_start = original_end

        return scene_slices

    def _observed_features(self, nodes_norm, valid_mask):
        """Build observed position-velocity features.

        Args:
            nodes_norm: [T, N, 2] normalized positions.
            valid_mask: [N] bool mask for observed-valid agents.

        Returns:
            features: [T_obs, Nv, 4] where channels are [x, y, vx, vy].
        """
        # positions: [T_obs, Nv, 2].
        positions = nodes_norm[:self.obs_length, valid_mask, :2]
        # velocities: [T_obs, Nv, 2].
        velocities = torch.zeros_like(positions)
        if self.obs_length > 1:
            # First velocity reuses the second-frame difference because there
            # is no t=-1 frame available.
            # positions[1:] - positions[:-1]: [T_obs - 1, Nv, 2].
            velocities[1:] = positions[1:] - positions[:-1]
            velocities[0] = velocities[1]
        # cat over coordinate channel: [T_obs, Nv, 2] + [T_obs, Nv, 2] -> [T_obs, Nv, 4].
        return torch.cat([positions, velocities], dim=-1)

    def _spatial_relation_prior(self, nodes_norm, nei_lists, valid_mask, scene_slices):
        """Compute STAR-style spatial prior for relation graph mixing.

        The prior uses last observed distance and the dataloader neighbor mask.
        It is block-diagonal by scene and row-max normalized, producing
        [Nv, Nv] values in roughly [0, 1]. ART learned relation weights are
        later mixed with this prior.
        """
        # current_pos: [Nv, 2], last observed position of valid agents.
        current_pos = nodes_norm[self.obs_length - 1, valid_mask, :2]
        num_nodes = current_pos.shape[0]
        # prior: [Nv, Nv].
        prior = current_pos.new_zeros(num_nodes, num_nodes)
        if num_nodes == 0:
            return prior

        # distance: [Nv, Nv], pairwise last-observed Euclidean distances.
        distance = torch.cdist(current_pos, current_pos, p=2)
        # distance_prior: [Nv, Nv], larger when agents are closer.
        distance_prior = torch.exp(-distance / max(self.spatial_sigma, 1e-6))

        # nei_lists is kept for compatibility with STAR dataloaders. For NBA
        # full-neighbor mode this is an all-to-all within-scene mask.
        if nei_lists.shape[0] > self.obs_length - 1:
            # First mask rows by valid agents, then mask columns by valid agents:
            # [T, N, N] -> [N, N] at last obs -> [Nv, Nv].
            neighbor_prior = nei_lists[self.obs_length - 1, valid_mask, :][:, valid_mask].float()
        else:
            neighbor_prior = torch.zeros_like(distance_prior)

        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 1:
                continue
            # scene_prior: [Ns, Ns]. The 0.25 term keeps a weak distance prior
            # even when the dataloader neighbor mask is zero.
            scene_prior = distance_prior[st:ed, st:ed] * (0.25 + neighbor_prior[st:ed, st:ed])
            eye = torch.eye(scene_nodes, dtype=torch.bool, device=current_pos.device)
            scene_prior = scene_prior.masked_fill(eye, 0.0)

            # Row-max normalization keeps the prior scale comparable to sigmoid
            # ART relation weights before the linear mixing step.
            # row_max: [Ns, 1], broadcasts across source-neighbor columns.
            row_max = scene_prior.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
            prior[st:ed, st:ed] = scene_prior / row_max
        return prior

    def _motion_context(self, nodes_norm, valid_mask, scene_slices):
        """Build compact motion/scene descriptors for adaptive gates.

        Returns:
            context: [Nv, 4] = [speed, acceleration, turning, scene_density].
        """
        current_pos = nodes_norm[self.obs_length - 1, valid_mask, :2]
        previous_pos = nodes_norm[self.obs_length - 2, valid_mask, :2]
        current_velocity = current_pos - previous_pos
        if self.obs_length > 2:
            previous_velocity = (
                nodes_norm[self.obs_length - 2, valid_mask, :2]
                - nodes_norm[self.obs_length - 3, valid_mask, :2]
            )
        else:
            previous_velocity = current_velocity

        # speed/acceleration: [Nv, 1].
        speed = current_velocity.norm(dim=-1, keepdim=True)
        acceleration = (current_velocity - previous_velocity).norm(dim=-1, keepdim=True)

        # turning: [Nv, 1], 0 means stable direction, 1 means strong direction change.
        cosine = F.cosine_similarity(current_velocity, previous_velocity, dim=-1, eps=1e-6).unsqueeze(-1)
        turning = (1.0 - cosine).clamp(0.0, 2.0) * 0.5

        # scene_density: [Nv, 1], normalized by a small ETH/UCY crowd scale.
        scene_density = current_pos.new_zeros(current_pos.size(0), 1)
        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 0:
                continue
            density_value = min(max(scene_nodes - 1, 0) / 10.0, 1.0)
            scene_density[st:ed] = density_value

        return torch.cat([speed, acceleration, turning, scene_density], dim=-1)

    def _adaptive_vite_adjacency(
        self,
        initial_nodes,
        motion_context,
        nodes_norm,
        nei_lists,
        valid_mask,
        scene_slices,
        use_spatial_gate=True,
    ):
        """Build adjacency for the adaptive ViTE branch.

        The branch starts from the strong vite_only uniform graph, then lets a
        learned gate choose how much STAR spatial prior should influence each
        target row.

        Returns:
            adjacency: [Nv, Nv].
            density: scalar kept-edge density.
            spatial_gate: [Nv, 1].
        """
        uniform_adj, _ = self._full_scene_adjacency(
            initial_nodes.size(0),
            scene_slices,
            initial_nodes.device,
            initial_nodes.dtype,
        )
        spatial_gate = initial_nodes.new_zeros(initial_nodes.size(0), 1)
        if not use_spatial_gate:
            return uniform_adj, self._adjacency_density(uniform_adj, scene_slices), spatial_gate

        spatial_prior = self._spatial_relation_prior(nodes_norm, nei_lists, valid_mask, scene_slices)
        spatial_adj, _ = self._row_normalized_adjacency(spatial_prior, scene_slices)
        # If a row has no useful spatial edge, fall back to the uniform ViTE row.
        spatial_row_sum = spatial_adj.sum(dim=-1, keepdim=True)
        spatial_adj = torch.where(spatial_row_sum.gt(0), spatial_adj, uniform_adj)

        # spatial_gate: [Nv, 1]. Higher value means stronger spatial-prior one-hop graph.
        spatial_gate = self.spatial_gate(torch.cat([initial_nodes, motion_context], dim=-1))
        adjacency = (1.0 - spatial_gate) * uniform_adj + spatial_gate * spatial_adj
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return adjacency, self._adjacency_density(adjacency, scene_slices), spatial_gate

    def _empty_aux(self, device):
        """Auxiliary outputs used when a batch has no valid observed agents."""
        return {
            "router_loss": torch.tensor(0.0, device=device),
            "aip_density": torch.tensor(0.0, device=device),
            "expert_usage": torch.zeros(2, device=device),
            "motion_gate_mean": torch.tensor(0.0, device=device),
            "spatial_gate_mean": torch.tensor(0.0, device=device),
        }

    def _adjacency_density(self, adjacency, scene_slices):
        """Return kept-edge density among possible within-scene non-self edges."""
        possible_edges = sum(max(ed - st, 0) * max(ed - st - 1, 0) for st, ed in scene_slices)
        if possible_edges > 0:
            return adjacency.gt(0).float().sum() / float(possible_edges)
        return adjacency.new_tensor(0.0)

    def _row_normalized_adjacency(self, weights, scene_slices):
        """Convert dense relation weights into row-normalized scene-block adjacency.

        This is the art_only path: it uses ART relation scores but disables
        adaptive top-p pruning, so every positive within-scene non-self edge can
        send messages.

        Args:
            weights: [Nv, Nv] dense relation weights.
            scene_slices: scene boundaries in compact valid-agent space.

        Returns:
            adjacency: [Nv, Nv] row-normalized dense adjacency.
            density: scalar kept-edge density.
        """
        adjacency = torch.zeros_like(weights)
        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 1:
                continue
            scene_weights = weights[st:ed, st:ed].clone()
            eye = torch.eye(scene_nodes, dtype=torch.bool, device=weights.device)
            scene_weights = scene_weights.masked_fill(eye, 0.0)
            scene_weights = scene_weights / scene_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            adjacency[st:ed, st:ed] = scene_weights
        return adjacency, self._adjacency_density(adjacency, scene_slices)

    def _full_scene_adjacency(self, num_nodes, scene_slices, device, dtype):
        """Build uniform full within-scene adjacency for the vite_only ablation.

        Returns:
            adjacency: [Nv, Nv], each non-singleton scene row sums to 1.
            density: scalar kept-edge density.
        """
        adjacency = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
        for st, ed in scene_slices:
            scene_nodes = ed - st
            if scene_nodes <= 1:
                continue
            block = torch.ones(scene_nodes, scene_nodes, device=device, dtype=dtype)
            eye = torch.eye(scene_nodes, dtype=torch.bool, device=device)
            block = block.masked_fill(eye, 0.0)
            block = block / float(scene_nodes - 1)
            adjacency[st:ed, st:ed] = block
        return adjacency, self._adjacency_density(adjacency, scene_slices)

    def forward(self, inputs, iftest=False):
        """Run one forward pass and produce K future candidates.

        Args:
            inputs: STAR-compatible 7-tuple:
                nodes_abs: [T, N, 2], absolute coordinates.
                nodes_norm: [T, N, 2], coordinates shifted by last obs frame.
                shift_value: [T, N, 2], unused by the model but kept for API compatibility.
                seq_list: [T, N], existence mask.
                nei_lists: [T, N, N], dataloader neighbor masks.
                nei_num: [T, N], unused compatibility field.
                batch_pednum: [num_scenes], scene sizes in original N space.
            iftest: Kept for compatibility; training/test share one forward path.

        Returns:
            predictions: [K, P, N, 2], normalized-coordinate futures.
            aux: auxiliary scalar/stat tensors for training logs and loss.
        """
        nodes_abs, nodes_norm, shift_value, seq_list, nei_lists, nei_num, batch_pednum = inputs
        device = nodes_norm.device
        # nodes_norm: [T, N, 2], so num_ped is N.
        num_ped = nodes_norm.shape[1]

        # Allocate full N-agent output first. Only observed-valid agents will
        # be filled; invalid agents remain zero and are ignored by loss masks.
        # predictions: [K, P, N, 2].
        predictions = nodes_norm.new_zeros(self.sample_num, self.pred_length, num_ped, 2)
        aux = self._empty_aux(device)

        # valid_mask: [N]. Nv = valid_mask.sum().
        valid_mask = self._valid_observed_mask(seq_list)
        if int(valid_mask.sum().item()) == 0:
            return predictions, aux

        # scene_slices uses compact Nv indexing, not original N indexing.
        scene_slices = self._scene_slices(batch_pednum, valid_mask)
        # observed_features: [T_obs, Nv, 4].
        observed_features = self._observed_features(nodes_norm, valid_mask)

        # [T_obs, Nv, 4] -> [T_obs, Nv, D].
        temporal_features = self.input_embedding(observed_features)
        # position_encoding keeps [T_obs, Nv, D].
        temporal_features = self.position_encoding(temporal_features)
        # temporal_encoder keeps [T_obs, Nv, D].
        temporal_features = self.temporal_encoder(temporal_features)
        # temporal_norm keeps [T_obs, Nv, D].
        temporal_features = self.temporal_norm(temporal_features)

        # Last observed temporal feature becomes the initial node feature [Nv, D].
        initial_nodes = temporal_features[-1]

        # Default aux values for ablations that do not activate AIP/router.
        router_loss = initial_nodes.new_tensor(0.0)
        aip_density = initial_nodes.new_tensor(0.0)
        expert_usage = initial_nodes.new_zeros(2)
        motion_gate_values = initial_nodes.new_zeros(initial_nodes.size(0), 1)
        spatial_gate_values = initial_nodes.new_zeros(initial_nodes.size(0), 1)
        motion_context = self._motion_context(nodes_norm, valid_mask, scene_slices)

        if self.ablation == "motion_only":
            # Motion-only baseline:
            #   T_obs trajectory encoder -> K decoders with constant-velocity prior.
            # No relation graph, no spatial prior, no AIP, no expert routing.
            pair_nodes = initial_nodes
            routed_nodes = initial_nodes
        elif self.ablation == "adaptive_motion_only":
            # New gate-only baseline:
            #   temporal encoder -> adaptive motion-prior decoder.
            # This isolates whether the learnable motion gate itself helps.
            pair_nodes = initial_nodes
            routed_nodes = initial_nodes
        else:
            # relation_features: [Nv, Nv, D]. It is zero when an ablation does
            # not use ART edge features, keeping the relational layer interface
            # unchanged.
            relation_features = initial_nodes.new_zeros(
                initial_nodes.size(0), initial_nodes.size(0), self.model_dim
            )

            if self.ablation in ("full", "no_motion_prior", "art_only", "art_aip"):
                # ART relation graph: W_art [Nv, Nv], R [Nv, Nv, D].
                relation_weights, relation_features = self.relation_graph(temporal_features, scene_slices)
            elif self.ablation == "spatial_only":
                # Spatial-only graph: use STAR/NBA/SDD distance-neighbor prior
                # as W, but keep ART relation edge features disabled.
                relation_weights = self._spatial_relation_prior(nodes_norm, nei_lists, valid_mask, scene_slices)
            elif self.ablation == "vite_only":
                # ViTE-only expert routing should not see ART or spatial prior.
                # Use a uniform full scene graph only as the one-hop expert's
                # message carrier.
                relation_weights, aip_density = self._full_scene_adjacency(
                    initial_nodes.size(0),
                    scene_slices,
                    initial_nodes.device,
                    initial_nodes.dtype,
                )
            elif self.ablation in ("adaptive", "adaptive_no_motion", "adaptive_no_spatial"):
                # Adaptive branch starts from ViTE, then optionally mixes a
                # learned spatial-prior row gate into the one-hop adjacency.
                relation_weights, aip_density, spatial_gate_values = self._adaptive_vite_adjacency(
                    initial_nodes,
                    motion_context,
                    nodes_norm,
                    nei_lists,
                    valid_mask,
                    scene_slices,
                    use_spatial_gate=(self.ablation != "adaptive_no_spatial"),
                )
            else:
                raise ValueError("Unknown ablation mode: {}".format(self.ablation))

            if self.ablation in ("full", "no_motion_prior"):
                spatial_prior = self._spatial_relation_prior(nodes_norm, nei_lists, valid_mask, scene_slices)
                # Mix learned temporal relation with STAR/NBA/SDD spatial prior before pruning.
                # relation_weights and spatial_prior are both [Nv, Nv].
                relation_weights = (
                    (1.0 - self.spatial_prior_mix) * relation_weights
                    + self.spatial_prior_mix * spatial_prior
                )
                # AIP returns sparse row-normalized adjacency A [Nv, Nv].
                adjacency, aip_density = self.interaction_pruning(relation_weights, scene_slices)
            elif self.ablation in ("spatial_only", "art_aip"):
                # Isolate the value of top-p pruning on either spatial prior
                # or ART relation weights.
                adjacency, aip_density = self.interaction_pruning(relation_weights, scene_slices)
            elif self.ablation == "art_only":
                # Use ART relation scores densely, without AIP.
                adjacency, aip_density = self._row_normalized_adjacency(relation_weights, scene_slices)
            elif self.ablation in ("vite_only", "adaptive", "adaptive_no_motion", "adaptive_no_spatial"):
                # relation_weights already stores the uniform adjacency.
                adjacency = relation_weights

            if self.ablation in ("vite_only", "adaptive", "adaptive_no_motion", "adaptive_no_spatial"):
                # No relational Transformer in the ViTE-only branch.
                pair_nodes = initial_nodes
            else:
                # Edge-aware relational Transformer updates local pairwise context.
                # pair_nodes starts [Nv, D] and every relational layer keeps [Nv, D].
                pair_nodes = initial_nodes
                for layer in self.relational_layers:
                    pair_nodes = layer(pair_nodes, relation_features, adjacency)

            if self.ablation in ("full", "no_motion_prior", "vite_only", "adaptive", "adaptive_no_motion", "adaptive_no_spatial"):
                # ViTE-style MoE: route between one-hop and virtual-node high-order experts.
                # router_weights: [Nv, 2]. router_loss: scalar. expert_usage: [2].
                router_weights, router_loss, expert_usage = self.router(pair_nodes)
                # one_hop_nodes/high_order_nodes: [Nv, D].
                one_hop_nodes = self.one_hop_expert(pair_nodes, adjacency)
                high_order_nodes = self.high_order_expert(pair_nodes, scene_slices)
                # router_weights[:, 0:1]: [Nv, 1], broadcasts over D.
                # weighted sum result routed_nodes: [Nv, D].
                routed_nodes = (
                    router_weights[:, 0:1] * one_hop_nodes
                    + router_weights[:, 1:2] * high_order_nodes
                )
                # Residual + LayerNorm keeps [Nv, D].
                routed_nodes = self.router_norm(routed_nodes + pair_nodes)
            else:
                # ART/spatial ablations stop before ViTE expert routing.
                routed_nodes = pair_nodes

        # Decoder sees raw temporal state, relation-updated state, and routed state.
        # cat over feature dim: [Nv, D] + [Nv, D] + [Nv, D] -> [Nv, 3D].
        final_features = torch.cat([initial_nodes, pair_nodes, routed_nodes], dim=-1)
        # current_pos: [Nv, 2].
        current_pos = nodes_norm[self.obs_length - 1, valid_mask, :2]
        # current_velocity: [Nv, 2].
        current_velocity = (
            nodes_norm[self.obs_length - 1, valid_mask, :2]
            - nodes_norm[self.obs_length - 2, valid_mask, :2]
        ) * self.velocity_scale

        motion_gate_for_decoder = None
        if self.motion_mode == "gated":
            # motion_gate_values: [Nv, 1], 0 means almost static baseline,
            # 1 means full constant-velocity baseline.
            motion_gate_values = self.motion_gate(torch.cat([final_features, motion_context], dim=-1))
            motion_gate_for_decoder = motion_gate_values

        # Stack K decoder heads: list of [P, Nv, 2] -> [K, P, Nv, 2].
        # Each decoder sees the same [Nv, 3D] features but has independent parameters.
        decoded = torch.stack(
            [
                decoder(
                    final_features,
                    current_pos,
                    current_velocity,
                    use_motion_prior=self.use_motion_prior,
                    motion_gate=motion_gate_for_decoder,
                )
                for decoder in self.decoder_heads
            ],
            dim=0,
        )

        # Scatter compact valid-agent predictions back into the original N-agent batch.
        # predictions[:, :, valid_mask, :]: [K, P, Nv, 2].
        # decoded: [K, P, Nv, 2].
        predictions[:, :, valid_mask, :] = decoded

        aux = {
            "router_loss": router_loss,
            "aip_density": aip_density.detach(),
            "expert_usage": expert_usage.detach(),
            "motion_gate_mean": motion_gate_values.mean().detach(),
            "spatial_gate_mean": spatial_gate_values.mean().detach(),
        }
        return predictions, aux

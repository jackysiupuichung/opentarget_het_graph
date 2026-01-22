import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index


class HGTConvRTE(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator with Relative Temporal Encoding (RTE).
    Based on standard HGTConv.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))
            
        # RTE: Temporal Linear Projection
        # We project the sine encoding (dim=dim) to dim to match k/v
        self.rte_lin = torch.nn.Linear(dim, dim) # Projects time encoding to K/V space

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        self.rte_lin.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _sine_encoding(self, t: Tensor, dim: int) -> Tensor:
        # t: [N]
        # returns [N, dim]
        device = t.device
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=device) * -emb)
        emb = t.unsqueeze(1).float() * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if dim % 2 == 1:
             emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_time_dict: Optional[Dict[EdgeType, Tensor]]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        time_emb_list: List[Tensor] = []
        
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])
            
            # RTE: Construct time encoding
            # If edge_time exists for this edge type
            if edge_time_dict and edge_type in edge_time_dict:
                # edge_time is [num_edges]
                # But k_dict[src] is [N_src]?
                # WAIT. HGT logic:
                # k_dict[src] is ALL nodes of input src type.
                # In standard HGT, we process ALL source nodes?
                # No, standard HGT iterates keys of edge_index_dict.
                # BUT `k_dict[src]` size depends on `x_dict`.
                # If we have multiple edge types with same src type, we reuse k_dict[src].
                
                # CRITICAL: HGT assumes we process all nodes of type S.
                # But edge_time is an EDGE attribute.
                # Constructing src_node_feat usually applies REL_EMB to NODE features.
                # It does NOT use edge-specific info until `propagate`?
                # `self.k_rel(ks, type_vec)`: `ks` are node features. `type_vec` aligns with the *edges*?
                # `_construct_src_node_feat` iterates over `edge_index_dict.keys()`?
                # Wait. `edge_index_dict` keys determine which relations we process.
                # `k_dict[src]` contains features for ALL source nodes involved?
                # No, `x_dict` contains all nodes.
                # The loop `for edge_type in edge_index_dict.keys()`:
                # Appends `k_dict[src]` to `ks`.
                # This seems to replicate Source Nodes for EACH relation they are involved in.
                
                # If we have edge-specific time, that time is on the EDGE (source -> target).
                # But `_construct_src_node_feat` prepares "Source Node Representations".
                # It does not yet know target nodes or specific edges (edge_index).
                # It prepares `k` and `v` matrices for message passing.
                # `k` size will be ` sum(N_src_nodes * Heads)`.
                # Wait, `N = k_dict[src].size(0)`. This IS the number of source nodes.
                # If we have edge timestamps, each edge has a time.
                # We can't assign a single time to a source node unless we aggregate?
                # OR, are we doing "Message Passing with Edge Attributes"?
                # If `edge_time` is an edge attribute, we should handle it in `message()`.
                
                # The PyG HGT implementation:
                # `k = self.k_rel(...)` -> `k` is (TotalSrcNodes * Heads, D).
                # `out = self.propagate(..., k=k)`
                # `propagate` takes `edge_index`.
                # In `message(k_j, ...)`: `k_j` is picked based on edge_index.
                
                # SO: If we want RTE dependent on EDGE time.
                # We must inject it in `message()`.
                # We cannot inject it in `_construct_src_node_feat` effectively unless we have node-level time.
                # User provided `edge_time`.
                # So we must pass `edge_time` to `propagate` and use it in `message`.
                
                pass 
            
            # (Continue structure matching)

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_time_dict: Optional[Dict[EdgeType, Tensor]] = None
    ) -> Dict[NodeType, Optional[Tensor]]:
        
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        
        # We pass None for edge_time_dict here because we handle it in propagate
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict, None)

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
            num_nodes=k.size(0))
            
        # construct_bipartite_edge_index merges edge_indices from different types.
        # It also merges 'edge_attr_dict' (p_rel) into 'edge_attr'.
        # We need to also merge 'edge_time_dict' into a single 'edge_times' tensor
        # aligned with 'edge_index'.
        
        edge_times = None
        if edge_time_dict:
            # We must replicate the logic of construct_bipartite_edge_index for edge_times
            # or manual concat.
            # construct_bipartite_edge_index iterates dict and concats. It respects order.
            # We can use the same utility if we handle it carefully, 
            # OR just manually concat since we know the order matches?
            # PyG's construct_bipartite_edge_index sorts keys?
            # "The usage of `construct_bipartite_edge_index` guarantees that edges are sorted..."
            # It iterates `edge_index_dict` keys.
            # We should iterate `edge_index_dict` keys and gather `edge_time`.
            
            e_times_list = []
            for edge_type in edge_index_dict.keys(): # Assumes insertion order preservation (Python 3.7+)
                if edge_type in edge_time_dict and edge_time_dict[edge_type] is not None:
                    e_times_list.append(edge_time_dict[edge_type])
                else:
                    # Provide dummy zeros if missing? or strict?
                    num_edges = edge_index_dict[edge_type].size(1)
                    e_times_list.append(torch.zeros(num_edges, device=k.device)) # TODO: Better default?
            
            if e_times_list:
                edge_times = torch.cat(e_times_list, dim=0)

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr, edge_times=edge_times)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor, 
                edge_times: Optional[Tensor],
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        
        # RTE: Add time encoding to k_j and v_j
        if edge_times is not None:
            # edge_times: [num_edges]
            # Encode
            D = q_i.size(-1)
            t_emb = self._sine_encoding(edge_times, D) # [num_edges, D]
            
            # Project time embedding to join the HGT latent space
            # Using self.rte_lin (D -> D)
            t_emb = self.rte_lin(t_emb) # [num_edges, D]
            
            # Add to Key and Value
            # Structure: k_j is [num_edges, Heads, D/Heads]?
            # Wait. k_j comes from k which is [N, H, D].
            # In message(), k_j is lifted to [num_edges, H, D].
            # q_i is [num_edges, H, D].
            # edge_attr is [num_edges, H, 1] usually? Or [num_edges, H]?
            # HGT: edge_attr comes from p_rel parameter which is [1, H].
            # construct_bipartite repeats it for edges.
            
            # We match dimensions.
            # t_emb is [num_edges, D_total]. Reshape to [num_edges, H, D_head].
            t_emb = t_emb.view(-1, self.heads, D)
            
            k_j = k_j + t_emb
            v_j = v_j + t_emb

        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')

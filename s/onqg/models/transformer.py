import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from onqg.utils.nn_utils import _get_activation_fn, _get_clones
from onqg.models.sublayer import MultiHeadedAttention


class DoubleAttnTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_sent, d_con=512, heads=8, d_ff=2048, dropout=0.1, att_drop=0.1, activation="relu", dual_enc=True):
        super(DoubleAttnTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(heads, d_model, query_dim=d_model, dropout=att_drop, use_structure=False)
        self.dual_enc = dual_enc
        self.d_sent = d_sent
        self.d_con = d_con
        self.sent_cross_attn = MultiHeadedAttention(
            heads, d_sent, query_dim=d_model, dropout=att_drop, use_structure=False
        )
        if self.d_sent != d_model and not dual_enc:
            self.kv_map = nn.Linear(self.d_sent, d_model)
        else:
            self.kv_map = None
        n_graph_head = 4 if self.d_con != 512 else 8
        if dual_enc:
            self.graph_cross_attn = MultiHeadedAttention(
                n_graph_head, self.d_con, query_dim=d_model, dropout=att_drop, use_structure=False
            )
            self.fuse_linear = nn.Linear(self.d_sent + self.d_con, d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Args:
            tgt (`FloatTensor`): set of `key_len`
                    key vectors `[batch, tgt_len, H]`
            memory (`FloatTensor`): set of `key_len`
                    key vectors `[batch, src_len, H]`
            tgt_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, tgt_len]`
            sent_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_sent_len]`
            graph_memory_mask: binary key2key mask indicating which keys have
                    non-zero attention `[tgt_len, src_graph_len]`
            tgt_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, tgt_len]`
            sent_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_sent_len]`
            graph_memory_key_padding_mask: binary padding mask indicating which keys have
                    non-zero attention `[batch, src_graph_len]`
            return:
            res:  [batch, tgt_len, H]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        sent_tgt2 = self.sent_cross_attn(
            tgt, sent_memory, sent_memory, mask=sent_memory_mask, key_padding_mask=sent_memory_key_padding_mask)[0]
        #print("sent_tgt2:", sent_tgt2.size())
        if self.dual_enc:
            graph_tgt2 = self.graph_cross_attn(
                sent_memory, graph_memory, graph_memory, mask=graph_memory_mask, key_padding_mask=graph_memory_key_padding_mask)[0]
            #print("graph_tgt2:", graph_tgt2.size())
            #tgt2 = self.fuse_linear(torch.cat([sent_tgt2, graph_tgt2], dim=-1))
        else:
            if self.kv_map is not None:
                sent_tgt2 = self.kv_map(sent_tgt2)
            tgt2 = sent_tgt2
        
        #tgt = tgt + self.dropout2(tgt2)
        #tgt = self.norm2(tgt)
        #tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        #tgt = tgt + self.dropout3(tgt2)
        #tgt = self.norm3(tgt)
        #print("tgt:", tgt.size())
        
        return graph_tgt2
        
class DoubleAttnTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(DoubleAttnTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        sent_memory: Tensor,
        graph_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        sent_memory_mask: Optional[Tensor] = None,
        graph_memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        sent_memory_key_padding_mask: Optional[Tensor] = None,
        graph_memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                sent_memory,
                graph_memory,
                tgt_mask=tgt_mask,
                sent_memory_mask=sent_memory_mask,
                graph_memory_mask=graph_memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                sent_memory_key_padding_mask=sent_memory_key_padding_mask,
                graph_memory_key_padding_mask=graph_memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output
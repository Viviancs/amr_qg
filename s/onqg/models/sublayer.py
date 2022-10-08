import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """
  Multi-Head Attention module from
  "Attention is All You Need"
  :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

  Similar to standard `dot` attention but uses
  multiple attention distributions simulataneously
  to select relevant items.

  Args:
     head_count (int): number of parallel heads
     model_dim (int): the dimension of keys/values/queries,
         must be divisible by head_count
     dropout (float): dropout parameter
  """

    def __init__(
        self,
        head_count,
        kv_dim,
        query_dim=512,
        dropout=0.1,
        use_structure=False,
        bias=True,
        alpha=1.0,
        beta=1.0,
    ):
        assert kv_dim % head_count == 0
        self.dim_per_head = kv_dim // head_count
        self.kv_dim = kv_dim
        self.query_dim = query_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(kv_dim, head_count * self.dim_per_head, bias=bias)
        self.linear_values = nn.Linear(kv_dim, head_count * self.dim_per_head, bias=bias)
        self.linear_query = nn.Linear(query_dim, head_count * self.dim_per_head, bias=bias)

        if use_structure:
            self.linear_structure_k = nn.Linear(self.dim_per_head, self.dim_per_head)
            self.linear_structure_v = nn.Linear(self.dim_per_head, self.dim_per_head)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(kv_dim, kv_dim)
        # self.final_linear = nn.Linear(kv_dim, query_dim)
        self.alpha = alpha
        self.beta = beta
        self.use_structure = use_structure
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.linear_query.weight)
        xavier_uniform_(self.linear_keys.weight)
        xavier_uniform_(self.linear_values.weight)

    def forward(
        self,
        query,
        key,
        value,
        structure=None,
        mask=None,
        key_padding_mask=None,
        layer_cache=None,
        type=None,
    ):
        """
        Compute the context vector and the attention vectors.

        Args:
        key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
        value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
        query (`FloatTensor`): set of `query_len`
                query vectors  `[batch, query_len, dim]`
        structure (`FloatTensor`): set of `query_len`
                query vectors  `[batch, query_len, query_len, dim]`

        mask: binary key2key mask indicating which keys have
                non-zero attention `[batch, key_len, key_len]`
        key_padding_mask: binary padding mask indicating which keys have
                non-zero attention `[batch, key_len]`
        
        Returns:
        (`FloatTensor`, `FloatTensor`) :
        * output context vectors `[batch, query_len, dim]`
        * one of the attention vectors `[batch, query_len, key_len]`
        """

        # CHECKS
        """
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.kv_dim % 8, 0)
        if mask is not None:
        batch_, q_len_, k_len_ = mask.size()
        aeq(batch_, batch)
        aeq(k_len_, k_len)
        aeq(q_len_ == q_len)
        print('q_len_mask: {}, q_len:{}'.format(q_len_, q_len))
        """
        # END CHECKS

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)
        # print('key_size', key.size())
        # print('value_size', value.size())

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:  # for decoder self-attn
            if type == "self":
                query, key, value = (  # [bsz, seq_len, H]
                    self.linear_query(query),
                    self.linear_keys(query),
                    self.linear_values(query),
                )
                if structure is not None and self.use_structure:
                    structure_k, structure_v = (
                        self.linear_structure_k(structure),  # [bsz, seq_len, seq_len, H]
                        self.linear_structure_v(structure),  # []
                    )
                else:
                    structure_k = None
                    structure_v = None

                key = shape(key)  # [bsz, nhead, key_len, H_head]
                value = shape(value)  # [bsz, nhead, value_len, H_head]

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat((layer_cache["self_keys"].to(device), key), dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat((layer_cache["self_values"].to(device), value), dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value

            elif type == "context":  # for decoder context-attn
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:  # encoder/decoder self/context attn
            #print(key.size())
            key = self.linear_keys(key)
            value = self.linear_values(value)
            # print('input:', query.size())
            # print('Linear:', self.linear_query)
            query = self.linear_query(query)
            if structure is not None and self.use_structure:
                structure_k, structure_v = (
                    self.linear_structure_k(structure),
                    self.linear_structure_v(structure),
                )
            else:
                structure_k = None
                structure_v = None

            key = shape(key)  # [batch_size, nhead, key_len, dim]
            value = shape(value)

        query = shape(query)  # [batch_size, nhead, key_len, dim]
        # print('key, query', key.size(), query.size())

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)  # attention scale
        scores = torch.matmul(query, key.transpose(2, 3))  # [batch_size, nhead, query_len, key_len]
        # print('scores', scores.size())

        if structure_k is not None:  # [batch_size, seq_len, seq_len, dim]
            q = query.transpose(1, 2)  # [batch_size, seq_len, nhead, dim]
            # print(q.size(), structure_k.transpose(2,3).size())
            scores_k = torch.matmul(
                q, structure_k.transpose(2, 3)
            )  # [batch_size, seq_len, nhead, seq_len]
            scores_k = scores_k.transpose(1, 2)  # [batch_size, nhead, seq_len, seq_len]
            # print (scores.size(),scores_k.size())
            scores = scores + self.alpha * scores_k

        if key_padding_mask is not None:  # padding mask
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # `[B, 1, 1, seq_len]`
            # print('key_padding_mask', key_padding_mask.size())
            scores = scores.masked_fill(key_padding_mask.bool(), -1e4)  # -1e4 allows fp16
            # print('scores_masked', scores)

        if mask is not None:  # key-to-key mask
            mask = mask.unsqueeze(1)  # `[B, 1, seq_len, seq_len]`
            scores = scores.masked_fill(mask.bool(), -1e4)  # -1e4 allows fp16

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        # print(drop_attn[0][0][3])
        context = torch.matmul(drop_attn, value)

        if structure_v is not None:
            drop_attn_v = drop_attn.transpose(1, 2)  # [batch_size, seq_len, nhead, seq_len]
            context_v = torch.matmul(
                drop_attn_v, structure_v
            )  # [batch_size, seq_len, seq_len, dim]
            context_v = context_v.transpose(1, 2)  # [batch_size, nhead, seq_len, dim]
            context = context + self.beta * context_v

        context = unshape(context)
        output = self.final_linear(context)

        # Return one attn
        first_head_attn = drop_attn.view(batch_size, head_count, query_len, key_len)[
            :, 0, :, :
        ].contiguous()
        return output, first_head_attn

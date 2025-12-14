import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from einops import repeat
from layers.Attn_Bias import BinaryAttentionBias
from layers.Attn_Projection import QueryKeyProjection, RotaryProjection
from utils.masking import TriangularCausalMask, TimerMultivariateMask, TimerCovariateMask
import torch.nn.functional as F




class FullAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        max_len: int = 100,
        covariate: bool = False,
        flash_attention: bool = False,
        mask_flag: bool = True,
        scale=None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()

        # --- core attention params ---
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        self.covariate = covariate
        self.flash_attention = flash_attention

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # ================
        # MTA CONFIG HERE
        # ================
        self.use_mta = True  # flip to False to get vanilla full attention

        # pre-softmax Q-K conv (Eq. 5)
        self.query_kernel_size = 3
        self.key_kernel_size = 3

        # post-softmax Q-K conv (Eq. 6) – disabled by default
        self.after_sm_query_kernel_size = 3
        self.after_sm_key_kernel_size = 3

        # head mixing conv (Sec. 3.2) – disabled by default
        self.head_kernel_size = 2      # set e.g. 2 to enable head conv
        self.pre_sm_linear_head = True   # linear head mixing before softmax
        self.post_sm_linear_head = True  # linear head mixing after softmax

        # padding mode along key dimension: "left", "right", or "both"
        self.pad_key = "both"
        self.mta_init_method = "normal"

        # ------- allocate MTA parameters -------
        if self.use_mta:
            # pre-softmax QK kernel: [H, 1, qdim, kdim]
            self.mta_kernel = None
            if self.query_kernel_size is not None and self.key_kernel_size is not None:
                self.mta_kernel = nn.Parameter(
                    torch.empty(
                        self.num_heads, 1,
                        self.query_kernel_size,
                        self.key_kernel_size,
                    )
                )

            # post-softmax QK kernel
            self.mta_kernel_after_sm = None
            if (
                self.after_sm_query_kernel_size is not None
                and self.after_sm_key_kernel_size is not None
            ):
                self.mta_kernel_after_sm = nn.Parameter(
                    torch.empty(
                        self.num_heads, 1,
                        self.after_sm_query_kernel_size,
                        self.after_sm_key_kernel_size,
                    )
                )

            # head conv kernel [G, c_h, c_h]
            self.head_kernel = None
            if self.head_kernel_size is not None:
                assert self.num_heads % self.head_kernel_size == 0
                self.head_kernel = nn.Parameter(
                    torch.empty(
                        self.num_heads // self.head_kernel_size,
                        self.head_kernel_size,
                        self.head_kernel_size,
                    )
                )

            # linear head mixing
            if self.pre_sm_linear_head:
                self.wpsm = nn.Linear(self.num_heads, self.num_heads, bias=False)
            if self.post_sm_linear_head:
                self.wposm = nn.Linear(self.num_heads, self.num_heads, bias=False)

            # initialize kernels / head mixing
            self.reset_mta_parameters()

    # ---------- helpers ----------

    def _build_numeric_mask(self, attn_mask, B, H, L, S, device, dtype):
        """
        Convert Timer's attn_mask (TriangularCausalMask or None or Tensor)
        into numeric mask [B,H,L,S] with values {0, -inf}.
        """
        if isinstance(attn_mask, TriangularCausalMask):
            # attn_mask.mask: [B, 1, L, S] bool
            bool_mask = attn_mask.mask.to(device)
            float_mask = torch.where(bool_mask, float("-inf"), 0.0).to(dtype)
            return float_mask.expand(B, H, L, S)

        elif attn_mask is None and self.mask_flag:
            base = torch.full((L, S), float("-inf"), device=device, dtype=dtype)
            base = torch.triu(base, diagonal=1)  # upper triangle = -inf, diag/below=0
            mask = base.unsqueeze(0).unsqueeze(0)  # [1,1,L,S]
            return mask.expand(B, H, L, S)

        elif isinstance(attn_mask, torch.Tensor):
            m = attn_mask.to(dtype)
            if m.dim() == 4 and m.size(1) == 1:
                m = m.expand(B, H, L, S)
            return m

        else:
            return None

    def _mta_convolution(
        self,
        scores: torch.Tensor,  # [B,H,L,S]
        mask: torch.Tensor,    # [B,H,L,S] with 0 / -inf
        kernel: torch.Tensor,  # [H, head_sz, qdim, kdim]
    ):
        # zero out masked positions so they don't affect conv
        scores = scores.clone()
        scores[mask == float("-inf")] = 0.0

        n_loc_heads, head_sz, qdim, kdim = kernel.shape
        assert n_loc_heads == self.num_heads

        # pad (L,S) dims: note F.pad order: (left,right,top,bottom)
        if self.pad_key == "left":
            scores_padded = F.pad(scores, (kdim - 1, 0, qdim - 1, 0), value=0.0)
        elif self.pad_key == "right":
            scores_padded = F.pad(scores, (0, kdim - 1, qdim - 1, 0), value=0.0)
        elif self.pad_key == "both":
            assert (kdim - 1) % 2 == 0
            scores_padded = F.pad(
                scores,
                ((kdim - 1) // 2, (kdim - 1) // 2, qdim - 1, 0),
                value=0.0,
            )
        else:
            raise ValueError(f"Unsupported pad_key: {self.pad_key}")

        # groups = num_heads / head_sz, like in the reference MTA code
        conv = F.conv2d(
            scores_padded,
            kernel,
            padding=0,
            groups=self.num_heads // head_sz,
        )
        del scores_padded

        return conv  # [B,H,L,S]

    def _head_convolution(self, scores, bsz, seq_len):
        # scores: [B, H, L, S]
        scores = scores.reshape(
            bsz,
            self.num_heads // self.head_kernel_size,
            self.head_kernel_size,
            seq_len,
            -1,
        )  # [B,G,c_h,L,S]
        scores_new = torch.empty_like(scores)
        for i in range(self.num_heads // self.head_kernel_size):
            # scores[:,i]: [B,c_h,L,S]
            scores_new[:, i] = torch.matmul(
                scores[:, i].transpose(1, -1),   # [B,S,L,c_h]
                self.head_kernel[i],            # [c_h,c_h]
            ).transpose(1, -1)                  # [B,c_h,L,S]

        scores = scores_new.reshape(bsz, self.num_heads, seq_len, -1)
        return scores

    def reset_mta_parameters(self):
        if not self.use_mta:
            return

        # q-k kernels
        if self.mta_kernel is not None:
            nn.init.kaiming_uniform_(self.mta_kernel, a=0.0)
        if self.mta_kernel_after_sm is not None:
            nn.init.kaiming_uniform_(self.mta_kernel_after_sm, a=0.0)

        # head conv: identity in each head group
        if self.head_kernel is not None:
            with torch.no_grad():
                g, b, c = self.head_kernel.shape
                assert b == c
                eye = torch.eye(b)
                self.head_kernel.copy_(eye.repeat(g, 1, 1))

        # linear head mixing
        if self.pre_sm_linear_head:
            nn.init.eye_(self.wpsm.weight)
        if self.post_sm_linear_head and hasattr(self, "wposm"):
            nn.init.eye_(self.wposm.weight)

    # ---------- main forward ----------

    def forward(self, queries, keys, values, attn_mask,
                n_vars=None, n_tokens=None, tau=None, delta=None):
        """
        queries: [B, L, H, E]
        keys:    [B, S, H, E]
        values:  [B, S, H, D]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert H == self.num_heads, "num_heads mismatch"

        scale = self.scale or 1.0 / sqrt(E)

        # [B,L,H,E] -> [B,H,L,E] etc.
        xq = queries.permute(0, 2, 1, 3).contiguous()
        xk = keys.permute(0, 2, 1, 3).contiguous()
        xv = values.permute(0, 2, 1, 3).contiguous()  # [B,H,S,D]

        # numeric mask [B,H,L,S] with 0 / -inf
        mask = self._build_numeric_mask(attn_mask, B, H, L, S, xq.device, xq.dtype)

        # ===== vanilla attention path if MTA off =====
        if not self.use_mta:
            scores = torch.einsum("bhle,bhse->bhls", xq, xk) * scale
            if mask is not None:
                scores = scores + mask
            A = torch.softmax(scores, dim=-1)
            A = self.dropout(A)
            out = torch.einsum("bhls,bhsd->blhd", A, xv)
            return (out.contiguous(), A) if self.output_attention else (out.contiguous(), None)

        # ======================
        # MTA ATTENTION PATH
        # ======================

        # raw scores: [B,H,L,S]
        scores = torch.matmul(xq, xk.transpose(2, 3)) * scale

        # pre-softmax q-k MTA (Eq. 5)
        if self.mta_kernel is not None:
            scores = self._mta_convolution(
                scores=scores,
                mask=mask if mask is not None else torch.zeros_like(scores),
                kernel=self.mta_kernel,
            )

        # pre-softmax head mixing (optional)
        if self.pre_sm_linear_head:
            scores = self.wpsm(scores.transpose(1, -1)).transpose(1, -1)

        # add mask and softmax
        if mask is not None:
            scores = scores + mask   # mask has 0 / -inf
        scores = torch.softmax(scores.float(), dim=-1).type_as(xq)  # [B,H,L,S]

        # post-softmax q-k MTA (Eq. 6)
        if self.mta_kernel_after_sm is not None:
            scores = self._mta_convolution(
                scores=scores,
                mask=mask if mask is not None else torch.zeros_like(scores),
                kernel=self.mta_kernel_after_sm,
            )
            if mask is not None:
                scores = torch.where(mask == float("-inf"), 0.0, scores)

        # post-softmax head mixing
        if self.head_kernel_size is not None:
            scores = self._head_convolution(scores=scores, bsz=B, seq_len=L)
        elif self.post_sm_linear_head:
            scores = self.wposm(scores.transpose(1, -1)).transpose(1, -1)

        # dropout on final attention weights
        A = self.dropout(scores)

        # aggregate values: [B,H,L,S] x [B,H,S,D] -> [B,L,H,D]
        out = torch.einsum("bhls,bhsd->blhd", A, xv)

        if self.output_attention:
            return out.contiguous(), A
        else:
            return out.contiguous(), None

class TimeAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, d_model=512, num_heads=8, max_len=100, covariate=False, flash_attention=False):
        super(TimeAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.covariate = covariate
        self.flash_attention = flash_attention
        self.qk_proj = QueryKeyProjection(dim=d_model, num_heads=num_heads, proj_layer=RotaryProjection, kwargs=dict(max_len=max_len),
                                          partial_factor=(0.0, 0.5),)
        self.attn_bias = BinaryAttentionBias(dim=d_model, num_heads=num_heads)

    def forward(self, queries, keys, values, attn_mask, n_vars, n_tokens, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # [B, H, L, E]
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        if self.flash_attention:
            values = values.permute(0, 2, 1, 3)

        seq_id = torch.arange(n_tokens * n_vars)
        seq_id = repeat(seq_id, 'n -> b h n', b=B, h=H)

        queries, keys = self.qk_proj(
            queries, keys, query_id=seq_id, kv_id=seq_id)

        scale = self.scale or 1. / sqrt(E)

        var_id = repeat(torch.arange(n_vars),
                        'C -> (C n_tokens)', n_tokens=n_tokens)
        var_id = repeat(var_id, 'L -> b h L', b=B, h=1).to(queries.device)

        attn_bias = self.attn_bias(var_id, var_id)

        if self.mask_flag:
            if attn_mask is None:
                if self.covariate:
                    attn_mask = TimerCovariateMask(
                        B, n_vars, n_tokens, device=queries.device)
                else:
                    attn_mask = TimerMultivariateMask(
                        B, n_vars, n_tokens, device=queries.device)
            attn_mask = attn_bias.masked_fill(attn_mask.mask, float("-inf"))
        else:
            attn_mask = attn_bias

        if self.flash_attention:
            V = torch.nn.functional.scaled_dot_product_attention(
                queries, keys, values, attn_mask)
        else:
            scores = torch.einsum("bhle,bhse->bhls", queries, keys)
            scores += attn_mask
            
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), None
        else:
            return V.contiguous(), None

class GatedGroupNorm(nn.Module):
    def __init__(self, n_heads, d_v):
        super().__init__()
        self.num_channels = n_heads * d_v
        # group by heads
        self.gn = nn.GroupNorm(num_groups=n_heads,
                               num_channels=self.num_channels,
                               affine=True)
        # simple learnable gate per channel
        self.gate = nn.Parameter(torch.ones(1, self.num_channels, 1))

    def forward(self, x):  # x: [B, L, H * d_v]
        B, L, C = x.shape      # C = H * d_v
        x = x.transpose(1, 2)  # [B, C, L]
        x = self.gn(x)         # [B, C, L]
        x = x * self.gate      # [B, C, L], same shape
        x = x.transpose(1, 2)  # [B, L, C]
        return x

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        self.d_keys = d_model // n_heads
        self.d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, self.d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, self.d_values * n_heads)
        self.out_projection = nn.Linear(self.d_values * n_heads, d_model)
        self.ggn = GatedGroupNorm(n_heads, self.d_values)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, n_vars=None, n_tokens=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            n_vars=n_vars,
            n_tokens=n_tokens,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, H * self.d_values)
        out = self.ggn(out)

        return self.out_projection(out), attn


import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def   __init__(self, input_dim, input_embedding_dim, dropout=0.1):
        super(SE, self).__init__()

        # 卷积层
        self.conv1x3 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv1x32 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv1x1 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.conv1x12 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, padding=0)
        self.conv1x13 = nn.Conv2d(input_dim, input_dim, kernel_size=(1, 1), stride=1, padding=0)

        # 线性层
        self.weight1 = nn.Linear(input_dim, input_dim)
        self.weight2 = nn.Linear(input_dim, input_dim)
        self.weight3 = nn.Linear(2 * input_dim, input_embedding_dim)
        self.weight4 = nn.Linear(input_dim, input_dim)
        self.weight5 = nn.Linear(input_dim, input_dim)
        self.weight6 = nn.Linear(input_dim, input_dim)
        self.weight7 = nn.Linear(input_dim, input_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def reverse_tensor(self, x):
        return torch.flip(x, dims=[3])  # 假设时间维度是第3维（从0开始）

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_embedding_dim)
        x = x.permute(0, 3, 2, 1)  # -> (B, input_embedding_dim, N, T)
        x_reversed = self.reverse_tensor(x)

        # === 分支1 ===
        x1 = self.conv1x3(x)
        x1 = x1.permute(0, 3, 2, 1)  # -> (B, T, N, C)
        x1 = F.relu(self.weight6(x1)) * F.sigmoid(self.weight7(x1))
        # x1 = F.gelu(self.weight6(x1)) * (self.weight7(x1))
        x1 = self.dropout(x1)  # dropout after gated activation

        # === 分支2 ===
        x2 = self.conv1x32(x_reversed)
        x2 = x2.permute(0, 3, 2, 1)  # -> (B, T, N, C)
        x2 = F.relu(self.weight4(x2)) * F.sigmoid(self.weight5(x2))
        # x2 = F.gelu(self.weight4(x2)) * (self.weight5(x2))
        x2 = self.dropout(x2)
        x2 = x2.permute(0, 3, 2, 1)  # -> (B, C, N, T)
        x2 = self.reverse_tensor(x2)
        x2 = x2.permute(0, 3, 2, 1)  # -> (B, T, N, C)

        # === 主分支 ===
        x = self.conv1x1(x)
         # dropout after conv
        x = x.permute(0, 3, 2, 1)  # -> (B, T, N, C)

        # === 加权融合 ===
        x2 = self.weight1(x1) + self.weight2(x2)
        x2 = self.dropout(x2)  # dropout after linear fusion

        out = torch.cat((x, x2), dim=-1)
        out = self.weight3(out)


        return out

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet1(nn.Module):
    def __init__(self, model_dim, kernel_size=2, dropout=0.1):
        super(TemporalConvNet1, self).__init__()
        dilation_size = 2
        padding = (kernel_size - 1) * dilation_size
        self.conv = nn.Conv2d(model_dim, model_dim, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
        self.chomp = Chomp1d(padding)
        # self.gelu =nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.tcn = nn.Sequential(self.conv, self.chomp, self.dropout)

    def forward(self, x):
        # -> (B, T, N, C)
        out = self.tcn(x.transpose(1, 3)).transpose(1, 3)
        out=torch.relu(out)
        out = self.tcn(out.transpose(1, 3)).transpose(1, 3)
        out = torch.tanh(out)
        return out


# class TCNmixer(nn.Module):
#     def __init__(
#             self, model_dim, feed_forward_dim, num_heads,  in_steps=12, dropout=0,mask=False
#     ):
#         super().__init__()
#         self.in_steps = in_steps
#         self.dropout1 = nn.Dropout(dropout)
#
#         self.weight1 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight2 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight3 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight4 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight5 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight6 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight11 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight12 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight13 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight14 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight15 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight16 = nn.Linear(in_features=model_dim, out_features=model_dim)
#         self.weight7 = nn.Linear(in_features=model_dim, out_features=model_dim)
#
#         self.att1=SelfAttentionLayertnn(model_dim, feed_forward_dim, num_heads, dropout)
#         self.att2 = SelfAttentionLayertnn(model_dim, feed_forward_dim, num_heads, dropout)
#
#     def forward(self, x):
#         # (batch_size, in_steps, num_nodes, input_embedding_dim)
#
#         # Slice the learnable parameters to match the 'steps' dimension
#
#         jigate = torch.sigmoid(self.weight1(self.att1(x)))
#         TCN2 = jigate * self.weight2(x) + (1 - jigate) * self.weight3(TCN2)
#
#         out = self.dropout1(TCN2)
#         out = torch.relu(out)
#         return out
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, input_embedding_dim, num_heads=8, mask=False):
        super().__init__()


        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = input_embedding_dim // num_heads

        self.FC_Q = nn.Linear(input_embedding_dim, input_embedding_dim)
        self.FC_K = nn.Linear(input_embedding_dim, input_embedding_dim)
        self.FC_V = nn.Linear(input_embedding_dim, input_embedding_dim)

        self.out_proj = nn.Linear(input_embedding_dim,input_embedding_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
class TCNAttentionLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.TCNcov1 = TemporalConvNet1(model_dim)


        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(self.TCNcov1(key))

        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
                             query @ key
                     ) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out
class TCNSelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attntcn = TCNAttentionLayer(model_dim, num_heads, mask)
        # self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attntcn(x, x, x)  # (batch_size, ..., length, model_dim)
        # out1 =self.attn(x,x,x)
        # out =self.weight1(out1)+self.weight2(outtcn)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
class SelfAttentionLayerem(nn.Module):
    def __init__(
        self,model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.weigh2 = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, adp_emb,dim=-2):
        x = x.transpose(dim, -2)
        adp_emb=adp_emb.transpose(dim, -2)
        adp_emb=self.weigh2(adp_emb)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x*adp_emb, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out
class SelfAttentionLayertnn(nn.Module):
    def __init__(
        self,model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()
        self.weigh1 = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.weigh2 = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.weigh3 = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.weigh4 = nn.Linear(in_features=model_dim, out_features=model_dim)
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.TCNcov22 = TemporalConvNet1(model_dim, kernel_size=2, dropout=0.1)
        self.TCNcov44 = TemporalConvNet1(model_dim, kernel_size=2, dropout=0.1)

    def forward(self, x):        # x: (batch_size, in_steps, num_nodes, input_embedding_dim)
        TCN2=self.TCNcov22(x)

        TCN2=self.ln1(x+TCN2)
        TCN4 = self.TCNcov44(x)

        TCN4=self.ln2(x+TCN4)
        out=F.sigmoid(self.weigh1(TCN2))*F.relu(self.weigh2(TCN4))
        # x = x.transpose(1, -2)
        # TCN2=TCN2.transpose(1, -2)
        #
        # out = self.attn(x, TCN2, x)
        # x = x.transpose(1, -2)
        # residual = x
        #   # (batch_size, ..., length, model_dim)
        # out = out.transpose(1, -2)
        # out = self.dropout1(out)
        # out = self.ln1(residual + out)
        #
        # residual = out
        # out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        # out = self.dropout2(out)
        #
        # out = self.ln2(residual + out)


        return out
class STAEformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        days_per_week=7,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=72,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,mask=False
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.days_per_week = days_per_week
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = ( tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim+input_embedding_dim
        )
        self.st_dim= ( tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
        )
        self.num_heads = num_heads
        self.weightt1 = nn.Linear(in_features=self.model_dim, out_features=self.model_dim)
        self.weightt2= nn.Linear(in_features=self.model_dim, out_features=self.model_dim)
        self.weightt3 = nn.Linear(in_features= 2*self.model_dim, out_features=self.model_dim)
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        # self.mixer = TCNmixer(
        #     input_embedding_dim,
        #     num_heads=8,
        #     dropout=0.1,
        #     in_steps=12,
        # )
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(days_per_week, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding_time = nn.Parameter(
                torch.randn(in_steps, num_nodes, adaptive_embedding_dim)
            )
            self.adaptive_embedding_space = nn.Parameter(
                torch.randn(in_steps, num_nodes, adaptive_embedding_dim)
            )
        else:
            self.adaptive_embedding_time = None
            self.adaptive_embedding_space = None

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(input_embedding_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(feed_forward_dim, input_embedding_dim),
        )
        self.ln1 = nn.LayerNorm(self.model_dim)
        self.ln2 = nn.LayerNorm(self.model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward2 = nn.Sequential(
            nn.Linear(input_embedding_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(feed_forward_dim, input_embedding_dim),
        )
        self.ln3 = nn.LayerNorm(self.model_dim)
        self.ln4 = nn.LayerNorm(self.model_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.attn_layers_t = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionLayerem(self.model_dim, feed_forward_dim, num_heads, dropout),  # 第一次用的
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout),

            ])
            for _ in range(num_layers)
        ])
        self.attn_layers_s = nn.ModuleList([
            nn.ModuleList([
                SelfAttentionLayerem(self.model_dim, feed_forward_dim, num_heads, dropout),  # 第一次用的
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            ])
            for _ in range(num_layers)
        ])
        self.mixers = nn.ModuleList(
            [TCNSelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout) for _ in range(num_layers)])
        # self.mixers = nn.ModuleList([
        #     nn.ModuleList([
        #         SelfAttentionLayerem(self.model_dim, feed_forward_dim, num_heads, dropout),  # 第一次用的
        #         TCNSelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
        #     ])
        #     for _ in range(num_layers)
        # ])
        self.se = SE(input_dim, input_embedding_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # 提取时间信息
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        x=self.se(x)
        # 投影输入维度
        # x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        # 准备时间与空间嵌入
        additional_features = []

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding((tod * self.steps_per_day).long())  # (B, T, N, D_tod)
            additional_features.append(tod_emb)

        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())  # (B, T, N, D_dow)
            additional_features.append(dow_emb)

        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )  # (B, T, N, D_spatial)
            additional_features.append(spatial_emb)

        # 拼接除 adaptive_embedding 以外的所有嵌入到 x
        if additional_features:
            x = torch.cat([x] + additional_features, dim=-1)  # (B, T, N, input_dim + D_other)
        if self.adaptive_embedding_dim > 0:
            adpt_emb = self.adaptive_embedding_time.expand(
                batch_size, *self.adaptive_embedding_time.shape
            )  # (B, T, N, D)
            adps_emb = self.adaptive_embedding_space.expand(
                batch_size, *self.adaptive_embedding_space.shape
            )  # (B, T, N, D)
        res=x
        for attn1, attn2 in self.attn_layers_t:
            xcnn = attn1(res, adpt_emb, dim=1)
            res = attn2(xcnn, dim=1)
        for attn1 in  self.mixers:
            x=attn1(x,dim=1)
        x=x+res

        for attn1, attn2 in self.attn_layers_s:

            xc = attn1(x, adps_emb, dim=2)
            x = attn2(xc, dim=2)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out


if __name__ == "__main__":
    model = STAEformer(207, 12, 12)
    summary(model, [64, 12, 207, 3])

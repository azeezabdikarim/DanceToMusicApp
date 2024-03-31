import math
import torch
import numpy as np
from torch import nn


class PoseRNN(nn.Module):
    def __init__(self, input_size=32*3, hidden_size=128, num_layers=3, output_size=256):
        super(PoseRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape input to be (B, N, input_size)
        x = x.view(x.size(0), x.size(1), -1)
        
        # LSTM forward pass
        x, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state for embedding
        embedding = self.fc(h_n[-1])
        return embedding
    
class Attention2D(nn.Module):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(c, nhead, dropout=dropout, bias=True, batch_first=True)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)
        if self_attn:
            kv = kv.unsqueeze(1).expand(-1, x.size(1), -1)
            kv = torch.cat([x, kv], dim=1)
        x = self.attn(x, kv, kv, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    
class LayerNorm1d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GlobalResponseNorm(nn.Module):
    "Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105"
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class ResBlock(nn.Module):
    def __init__(self, c, c_skip=None, kernel_size=3, dropout=0.0):
        super().__init__()
        # self.depthwise = nn.Conv1d(c + c_skip if c_skip is not None else c, c, 
        #                            kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.depthwise = nn.Conv2d(c + c_skip if c_skip is not None else c, c, 
                                   kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = LayerNorm1d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            # Crop x to the size of x_skip if they don't match
            if x.shape[2] > x_skip.shape[2]:
                x = x[..., :x_skip.shape[2], :x_skip.shape[3]]
            elif x.shape[2] < x_skip.shape[2]:
                x_skip = x_skip[..., :x.shape[2], :x.shape[3]]
            x = torch.cat([x, x_skip], dim=1)
        x = self.norm(self.depthwise(x)).permute(0, 2, 3, 1)
        x = self.channelwise(x).permute(0, 3, 1, 2)
        return x + x_res

    
class AttnBlock(nn.Module):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm1d(c, elementwise_affine=False, eps=1e-6)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_cond, c)
        )

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x
    
class FeedForwardBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm1d(c, elementwise_affine=False, eps=1e-6)
        self.channelwise = nn.Sequential(
            nn.Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(dropout),
            nn.Linear(c * 4, c)
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class TimestepBlock(nn.Module):
    def __init__(self, c, c_timestep):
        super().__init__()
        self.mapper = nn.Linear(c_timestep, c * 2)

    def forward(self, x, t):
        a, b = self.mapper(t)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + a) + b


class Dance2MusicDiffusion(nn.Module):
    def __init__(self, c_in=256, c_out=256, num_labels=8192, c_r=64, patch_size=2, c_cond=1024, pose_seq_len=72, num_keypoints = 96,
                 c_hidden=[640, 1280, 1280], nhead=[-1, 16, 16], blocks=[6, 16, 6], level_config=['CT', 'CTA', 'CTA'],
                 dance_embd=1024, clip_seq_len=4, kernel_size=3, dropout=0.1, self_attn=True):
        super().__init__()
        self.c_r = c_r
        self.c_cond = c_cond
        self.num_labels = num_labels
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)

        # CONDITIONING
        # self.dance_conv1d = nn.Conv1d(in_channels=num_keypoints, out_channels=dance_embd, kernel_size=5, padding=2)
        self.dance_embdeder = PoseRNN(hidden_size=128, num_layers=1, output_size=c_cond, input_size=num_keypoints)
        self.dance_mapper = nn.Linear(dance_embd, c_cond)
        self.seq_norm = nn.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6)

        self.in_mapper = nn.Sequential(
            nn.Embedding(num_labels, c_in),
            nn.LayerNorm(c_in, elementwise_affine=False, eps=1e-6)
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(c_in , c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i in range(len(c_hidden)):
            down_block = nn.ModuleList()
            if i > 0:
                down_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    nn.Conv2d(c_hidden[i - 1], c_hidden[i], kernel_size=(2,1), stride=(2,1)),
                ))
            for _ in range(blocks[i]):
                for block_type in level_config[i]:
                    down_block.append(get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i]))
            self.down_blocks.append(down_block)

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            up_block = nn.ModuleList()
            for j in range(blocks[i]):
                for k, block_type in enumerate(level_config[i]):
                    up_block.append(get_block(block_type, c_hidden[i], nhead[i],
                                              c_skip=c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0,
                                              dropout=dropout[i]))
            if i > 0:
                up_block.append(nn.Sequential(
                    LayerNorm2d(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    nn.ConvTranspose2d(c_hidden[i], c_hidden[i - 1], kernel_size=(2,1), stride=(2,1)),
                ))
            self.up_blocks.append(up_block)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_hidden[0], c_out, kernel_size=1),
        )
        self.out_mapper = nn.Sequential(
            LayerNorm2d(c_out, elementwise_affine=False, eps=1e-6),
            nn.Conv2d(c_out, num_labels, kernel_size=1, bias=False)
        )

        # --- WEIGHT INIT ---
        self.apply(self._init_weights)  # General init
        nn.init.normal_(self.dance_mapper.weight, std=0.02)
        # nn.init.normal_(self.clip_mapper.weight, std=0.02)
        # nn.init.normal_(self.clip_image_mapper.weight, std=0.02)
        torch.nn.init.xavier_uniform_(self.embedding[0].weight, 0.02)
        nn.init.constant_(self.clf[1].weight, 0)
        nn.init.normal_(self.in_mapper[0].weight, std=np.sqrt(1 / num_labels))
        self.out_mapper[-1].weight.data = self.in_mapper[0].weight.data[:, :, None, None].clone()

        for level_block in self.down_blocks + self.up_blocks:
            for block in level_block:
                if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                    block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks))
                elif isinstance(block, TimestepBlock):
                    nn.init.constant_(block.mapper.weight, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def gen_t_embedding(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    # def gen_c_embeddings(self, byt5, clip, clip_image):
    #     seq = self.byt5_mapper(byt5)
    #     if clip is not None:
    #         clip = self.clip_mapper(clip).view(clip.size(0), -1, self.c_cond)
    #         seq = torch.cat([seq, clip], dim=1)
    #     if clip_image is not None:
    #         clip_image = self.clip_image_mapper(clip_image).view(clip_image.size(0), -1, self.c_cond)
    #         seq = torch.cat([seq, clip_image], dim=1)
    #     seq = self.seq_norm(seq)
    #     return seq

    def _down_encode(self, x, r_embed, c_embed):
        level_outputs = []
        for i, down_block in enumerate(self.down_blocks):
            for block in down_block:
                if isinstance(block, ResBlock):
                    x = block(x)
                elif isinstance(block, AttnBlock):
                    x = block(x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, c_embed):
        x = level_outputs[0]
        for i, up_block in enumerate(self.up_blocks):
            for j, block in enumerate(up_block):
                if isinstance(block, ResBlock):
                    x = block(x, level_outputs[i] if j == 0 and i > 0 else None)
                elif isinstance(block, AttnBlock):
                    x = block(x, c_embed)
                elif isinstance(block, TimestepBlock):
                    x = block(x, r_embed)
                else:
                    x = block(x)
        return x

    def forward(self, x, t, pose_sequence, clip=None, clip_image=None, x_cat=None):
        if x_cat is not None:
            x = torch.cat([x, x_cat], dim=1)
        # Process the conditioning embeddings
        t_embed = self.gen_t_embedding(t)
        # dance_embed = dance_embed.transpose(1, 2)
        d_embed = self.dance_embdeder(pose_sequence)
        # d_embed = self.dance_mapper(dance_embed)

        # Model Blocks
        # make x a torch tensor of integers
        x = self.embedding(self.in_mapper(x.long()).permute(0, 3, 1, 2))
        level_outputs = self._down_encode(x, t_embed, d_embed)
        x = self._up_decode(level_outputs, t_embed, d_embed)
        x = self.out_mapper(self.clf(x))

        probabilities = nn.functional.softmax(x, dim=-1)  # Convert logits to probabilities
        log_probabilities = torch.log(probabilities)
        return x, log_probabilities

    def add_noise(self, x, t, mask=None, random_x=None):
        if mask is None:
            mask = (torch.rand_like(x.float()) <= t[:, None, None]).long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask
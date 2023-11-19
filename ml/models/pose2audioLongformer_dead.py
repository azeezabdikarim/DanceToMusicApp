import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        #Linear Layors for K, V, Q
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        B = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(B, value_len, self.heads, self.head_dim)
        keys = keys.reshape(B, key_len, self.heads, self.head_dim)
        queries = query.reshape(B, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])
        # queries shape: (B, query_len, heads, head_dim)
        # keys shape: (B, key_len, heads, head_dim)
        # energy shape: (B, heads, query_len, key_len)

        if mask is not None:
            mask_expanded = mask.expand_as(energy)
            energy = energy.masked_fill(mask_expanded == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            B, query_len, self.heads*self.head_dim
        )
        # attention shape : (N, heads, query_len, key_len)
        # values shape : (N, value_len, heads, heads_dim)
        # out shape : (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out
    
class LongformerSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, window_size, use_global_token=True):
        super(LongformerSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.window_size = window_size
        self.use_global_token = use_global_token

        #Linear Layors for K, V, Q
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        B = query.shape[0]
        value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]

        # Split embedding into self.heads pieces
        values = values.reshape(B, value_len, self.heads, self.head_dim)
        keys = keys.reshape(B, key_len, self.heads, self.head_dim)
        queries = query.reshape(B, query_len, self.heads, self.head_dim)

        # Normal attention computation for global token
        if self.use_global_token:
            global_queries = queries[:, 0:1, :, :]
            global_energy = torch.einsum("bqhd,bkhd->bhqk", [global_queries, keys])
            global_attention = torch.softmax(global_energy / (self.embed_size ** (1/2)), dim=3)
            global_out = torch.einsum("bhql,blhd->bqhd", [global_attention, values])

        # Window-based attention for remaining tokens
        window_queries = queries[:, 1 if self.use_global_token else 0:, :, :]
        energies = torch.zeros(B, self.heads, window_queries.shape[1], key_len, device=queries.device)

        for i in range(window_queries.shape[1]):
            # Define the attention window for each query token
            start = max(i - self.window_size // 2, 0)
            end = min(i + self.window_size // 2 + 1, key_len)

            # Calculate energy within the window
            local_energy = torch.einsum("bqhd,bkhd->bhqk", [window_queries[:, i:i+1, :, :], keys[:, start:end, :, :]])
            energies[:, :, i, start:end] = local_energy.squeeze(2)

        # Apply mask and softmax
        if mask is not None:
            energies = energies.masked_fill(mask == 0, float("-1e20"))
        attention = torch.softmax(energies / (self.embed_size ** (1/2)), dim=3)

        local_out = torch.einsum("bhql,blhd->bqhd", [attention, values])

        if self.use_global_token:
            out = torch.cat([global_out, local_out], dim=1)
        else:
            out = local_out

        out = out.reshape(B, out.shape[1], self.heads * self.head_dim)

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, window_size=32, use_global_token=True):
        super(DecoderTransformerBlock, self).__init__()
        self.attention = LongformerSelfAttention(embed_size, heads, window_size, use_global_token)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention.unsqueeze(1) + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self,
            # src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
            ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        # self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def positional_encoding(self, seq_len):
        pe = torch.zeros(seq_len, self.embed_size).to(self.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask):
        B, seq_length, _ = x.shape
        pos_encoding = self.positional_encoding(seq_length)

        out = self.dropout(x + pos_encoding[:, :seq_length, :])
        # out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device, window_size=32, use_global_token=True):
        super(DecoderBlock, self).__init__()
        # self.attention = SelfAttention(embed_size, heads)
        self.attention = LongformerSelfAttention(embed_size, heads, window_size, use_global_token)
        self.norm = nn.LayerNorm(embed_size)
        self.deivce = device
        self.transformer_block = DecoderTransformerBlock(
            embed_size, heads, dropout, forward_expansion, window_size, use_global_token
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x,x,x,trg_mask)
        query = self.dropout(self.norm(attention.unsqueeze(1) + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
                code_book_len,
                embed_size,
                num_layers,
                heads,
                forward_expansion,
                dropout,
                device,
                max_length,
                window_size,
                use_global_token=True
                ):
        super(Decoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.codebook_embedding = nn.Embedding(code_book_len, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device,
                    window_size,
                    use_global_token
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, code_book_len)
        self.dropout = nn.Dropout(dropout)

    def positional_encoding(self, seq_len):
        pe = torch.zeros(seq_len, self.embed_size).to(self.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_size, 2).float() * (-math.log(10000.0) / self.embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, enc_out, src_mask, trg_mask):
        B, _, seq_length = x.shape
        positions = self.positional_encoding(seq_length)
        x = x.long()
        x = self.dropout(self.codebook_embedding(x) + positions[:, :seq_length, :])

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        
        # Applying softmax to get probabilities
        softmax_out = F.softmax(out, dim=-1)
        if softmax_out.shape[1] > 1:  
            final_output_softmax = softmax_out[:, 1:, :]
            logits_out = out[:, 1:, :]
            offset = 1
        else:
            final_output_softmax = softmax_out # special situation when start token is the only value
            logits_out = out
            offset = 0 

        # Finding the index with maximum probability
        argmax_out = torch.argmax(softmax_out, dim=-1)
        final_output_argmax = argmax_out[:, 1:]     # Remove the start token from the output

        
        return softmax_out, argmax_out, offset, logits_out

class Pose2AudioLongformer(nn.Module):
    def __init__(
        self,
        # src_vocab_size,
        code_book_len,
        src_pad_idx,
        trg_pad_idx,
        embed_size=50,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="mps",
        max_length=2000,
        window_size=32,
        use_global_token=True
    ):
        super(Pose2AudioLongformer, self).__init__()
        self.encoder = Encoder(
            # src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            code_book_len,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
            window_size,
            use_global_token
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.window_size = window_size
        self.use_global_token = use_global_token
        self.device = device
        self.start_token_code = torch.zeros(1, dtype=torch.long).to(device)  # Add this line, assuming the start token is represented by the integer 0

    def make_src_mask(self, src):
        # Check if the last two dimensions are all zeros
        src_mask = (src.sum(dim=[-1, -2]) != 0)
        # Expand the dimensions to match the attention mask shape
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def forward(self, src, trg, src_mask = None):
        # if src_mask is None:
        #     src_mask = self.make_src_mask(src)
        # start_tokens = self.start_token_code.repeat(trg.shape[0], 1)  # Repeat the start token for each item in the batch
        # trg_with_start = torch.cat((start_tokens, trg), dim=1)

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_longformer_trg_mask(trg, self.window_size, self.use_global_token)
        # trg_mask = self.make_trg_mask(trg)
        # trg_mask = self.make_trg_mask(trg_with_start)

        B, N, _, _ = src.shape
        enc_src = self.encoder(src.view(B, N, -1), src_mask)
        out = self.decoder(trg, enc_src.unsqueeze(1), src_mask, trg_mask)
        return out

    def make_trg_mask(self, trg):
        B, seq_length = trg.shape
        trg_mask = torch.tril(torch.ones((seq_length, seq_length))).expand(
            B, 1, seq_length, seq_length
        )
        return trg_mask.to(self.device)
    
    def make_longformer_trg_mask(self, trg, window_size, use_global_token=True):
        B, _, seq_length = trg.shape
        trg_mask = torch.zeros((B, 1, seq_length, seq_length), device=self.device)

        # Mask for global token attending to all tokens and vice versa
        if use_global_token:
            trg_mask[:, :, 0, :] = 1
            trg_mask[:, :, :, 0] = 1

        # Create a 1D window mask for a single sequence
        window_mask = torch.zeros(seq_length, seq_length, device=self.device)
        for i in range(seq_length):
            start = max(i - window_size // 2, 0)
            end = min(i + window_size // 2 + 1, seq_length)
            window_mask[i, start:end] = 1

        # Remove the first row if we're using a global token, as it has a different pattern
        if use_global_token:
            window_mask = window_mask[1:, :]

        # Expand dimensions to make it compatible with trg_mask
        window_mask_expanded = window_mask.unsqueeze(0).unsqueeze(1)

        # Tile across the batch dimension
        window_mask_tiled = window_mask_expanded.repeat(B, 1, 1, 1)

        # Assign to the appropriate slice of trg_mask
        trg_mask[:, :, (1 if use_global_token else 0):, :] = window_mask_tiled

        return trg_mask


    def generate(self, src, src_mask, max_length=100, temperature=1.0):
        B, N, _, _ = src.shape
        enc_src = self.encoder(src.view(B, N, -1), src_mask)

        # Initialize with a single start token for each sequence in the batch
        trg = self.start_token_code.repeat(B, 1).to(self.device)
        
        generated_tokens = []

        for i in range(max_length):
            trg_mask = self.make_longformer_trg_mask(trg)
            output_softmax, output_argmax, offset, logits = self.decoder(trg, enc_src, src_mask, trg_mask)
            
            # Apply temperature to logits and re-normalize to probabilities
            output_softmax = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            
            # Sample from the softmax output
            next_token = torch.multinomial(output_softmax, 1).squeeze(-1)

            # Store the generated tokens
            generated_tokens.append(next_token.unsqueeze(1))

            # Update the target sequence for the next iteration
            trg = torch.cat((trg, next_token.unsqueeze(1)), dim=1)

        # Concatenate along sequence length dimension to get final output
        final_trg = torch.cat(generated_tokens, dim=1)

        return final_trg






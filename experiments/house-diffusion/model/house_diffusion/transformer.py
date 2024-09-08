import math
import typing
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import timestep_embedding

import numpy as np

def dec2bin(xinp, bits):
    mask = 2 ** th.arange(bits - 1, -1, -1).to(xinp.device, xinp.dtype)
    return xinp.unsqueeze(-1).bitwise_and(mask).ne(0).float()

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = th.arange(max_len).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = th.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = th.sin(position * div_term)
        pe[0, :, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[0:1, :x.size(1)]
        return self.dropout(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = activation
    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = th.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 1, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    
    output = th.matmul(scores, v)
    
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)# calculate attention using function we will define next
        
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.door_attn = MultiHeadAttention(heads, d_model)
        self.gen_attn = MultiHeadAttention(heads, d_model)
        
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, door_mask, self_mask, gen_mask):
        assert (gen_mask.max()==1 and gen_mask.min()==0), f"{gen_mask.max()=}, {gen_mask.min()=}"

        x2 = self.norm_1(x)

        x = x + self.dropout(self.door_attn(x2, x2, x2, door_mask)) \
                + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
                + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))

        x2 = self.norm_2(x)

        x = x + self.dropout(self.ff(x2))

        return x


class WallEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)
        
        self.wall_attn = MultiHeadAttention(heads, d_model)
        
        
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, wall_points, wall_padding_mask):
        assert (wall_padding_mask.max()==1 and wall_padding_mask.min()==0), f"{wall_padding_mask.max()=}, {wall_padding_mask.min()=}"

        x = wall_points

        x2 = self.norm_1(x)

        x = x + self.dropout(self.wall_attn(x2, x2, x2, wall_padding_mask)) \

        x2 = self.norm_2(x)

        x = x + self.dropout(self.ff(x2))

        return x


class EncoderLayerWithStructuralCrossAttention(nn.Module):
    def __init__(self, d_model, heads, dropout, activation):
        super().__init__()
        self.norm_1 = nn.InstanceNorm1d(d_model)
        self.norm_2 = nn.InstanceNorm1d(d_model)

        self.norm_door_type = nn.InstanceNorm1d(d_model)

        self.norm_struct = nn.InstanceNorm1d(d_model)
        
        self.self_attn = MultiHeadAttention(heads, d_model)
        self.door_attn = MultiHeadAttention(heads, d_model)
        self.gen_attn = MultiHeadAttention(heads, d_model)

        self.struct_attn = MultiHeadAttention(heads, d_model)
        
        self.ff = FeedForward(d_model, d_model*2, dropout, activation)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, self_mask, gen_mask, x_structural, structural_mask, door_only_mask, passage_ony_mask, entrance_only_mask, door_type_emb):
        assert (gen_mask.max()==1 and gen_mask.min()==0), f"{gen_mask.max()=}, {gen_mask.min()=}"

        x2 = self.norm_1(x)

        x2_structural = self.norm_struct(x_structural)

        door_type_emb = self.norm_door_type(door_type_emb)

        x = x \
                + self.dropout(self.self_attn(x2, x2, x2, self_mask)) \
                + self.dropout(self.gen_attn(x2, x2, x2, gen_mask))
        
        x_door_only_key_value = x2 + door_type_emb[0, :]
        x_passage_only_key_value = x2 + door_type_emb[1, :]
        x_entrance_only_key_value = x2 + door_type_emb[2, :]

        x += self.dropout(self.door_attn(x2, x_door_only_key_value, x_door_only_key_value, door_only_mask))
        x += self.dropout(self.door_attn(x2, x_passage_only_key_value, x_passage_only_key_value, passage_ony_mask))
        x += self.dropout(self.door_attn(x2, x_entrance_only_key_value, x_entrance_only_key_value, entrance_only_mask))
        
        x = x + self.dropout(self.struct_attn(x2, x2_structural, x2_structural, structural_mask))

        x2 = self.norm_2(x)

        x = x + self.dropout(self.ff(x2))

        return x

class TransformerModel(nn.Module):
    """
    The full Transformer model with timestep embedding.
    """

    def __init__(
        self,
        in_channels: int,
        condition_channels: int,
        model_channels: int,
        out_channels: int,
        dataset,
        use_checkpoint,
        use_unet,
        analog_bit,
        struct_in_channels: int,
        img_size: int=512,
        use_wall_self_attention: bool=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.condition_channels = condition_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.time_channels = model_channels
        self.use_checkpoint = use_checkpoint
        self.analog_bit = analog_bit
        self.use_unet = use_unet
        self.num_transfomers_layers = 4

        self.struct_in_channels = struct_in_channels

        self.use_structural_cross_attention = struct_in_channels > 0
        self.use_wall_self_attention = use_wall_self_attention

        self.img_size = img_size
        self.num_bits = int(np.log2(self.img_size))

        print(f"Created model with {self.use_structural_cross_attention=} and {self.struct_in_channels=}")


        # self.pos_encoder = PositionalEncoding(model_channels, 0.001)
        # self.activation = nn.SiLU()
        self.activation = nn.ReLU()

        self.time_embed = nn.Sequential(
            nn.Linear(self.model_channels, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.time_channels),
        )
        
        self.input_emb = nn.Linear(self.in_channels, self.model_channels)
        self.condition_emb = nn.Linear(self.condition_channels, self.model_channels)

        if self.use_structural_cross_attention:
            # This isn't used in the original EncoderLayer
            self.door_type_emb = nn.Linear(3, self.model_channels)

        if use_unet:
            raise ValueError("Expected use_unet to be False")


        if self.use_structural_cross_attention:
            self.struct_in_channels: int = self.struct_in_channels # type: ignore

            self.transformer_layers: typing.List[EncoderLayerWithStructuralCrossAttention] = nn.ModuleList([EncoderLayerWithStructuralCrossAttention(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_transfomers_layers)]) # type: ignore

            self.struct_emb = nn.Linear(self.struct_in_channels, self.model_channels)

            if self.use_wall_self_attention:
                self.wall_self_attention_layers = nn.ModuleList([WallEncoderLayer(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_transfomers_layers)]) # type: ignore

        else:
            self.transformer_layers: typing.List[EncoderLayer] = nn.ModuleList([EncoderLayer(self.model_channels, 4, 0.1, self.activation) for x in range(self.num_transfomers_layers)]) # type: ignore
        # self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(self.model_channels, 4, self.model_channels*2, 0.1, self.activation, batch_first=True) for x in range(self.num_layers)])

        self.output_linear1 = nn.Linear(self.model_channels, self.model_channels)
        self.output_linear2 = nn.Linear(self.model_channels, self.model_channels//2)
        self.output_linear3 = nn.Linear(self.model_channels//2, self.out_channels)

        if not self.analog_bit:
            self.output_linear_bin1 = nn.Linear(2 * 9 + 2 * 9 * self.num_bits + self.model_channels, self.model_channels)

            if self.use_structural_cross_attention:
                self.output_linear_bin2 = EncoderLayerWithStructuralCrossAttention(self.model_channels, 1, 0.1, self.activation)
                self.output_linear_bin3 = EncoderLayerWithStructuralCrossAttention(self.model_channels, 1, 0.1, self.activation)

                self.struct_linear_bin1 = nn.Linear(2 * 9 + 2 * 9 * self.num_bits + self.model_channels, self.model_channels)
            else:
                self.output_linear_bin2 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
                self.output_linear_bin3 = EncoderLayer(self.model_channels, 1, 0.1, self.activation)
            
            self.output_linear_bin4 = nn.Linear(self.model_channels, 2 * self.num_bits)

        print(f"Number of model parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    
    def expand_points(self, points: th.Tensor, connections: th.Tensor) -> th.Tensor:
        """Probably this is the AU augmentation function that samples 
        (according to the paper L=8) points from the corner along the wall to the next corner, 
        and concatenates them to the original point."""

        def average_points(point1, point2):
            points_new = (point1+point2)/2
            return points_new
        
        p1 = points
        p1 = p1.view([p1.shape[0], p1.shape[1], 2, -1])
        p5 = points[th.arange(points.shape[0])[:, None], connections[:,:,1].long()]
        p5 = p5.view([p5.shape[0], p5.shape[1], 2, -1])
        p3 = average_points(p1, p5)
        p2 = average_points(p1, p3)
        p4 = average_points(p3, p5)
        p1_5 = average_points(p1, p2)
        p2_5 = average_points(p2, p3)
        p3_5 = average_points(p3, p4)
        p4_5 = average_points(p4, p5)
        points_new = th.cat((p1.view_as(points), p1_5.view_as(points), p2.view_as(points),
            p2_5.view_as(points), p3.view_as(points), p3_5.view_as(points), p4.view_as(points), p4_5.view_as(points), p5.view_as(points)), 2)
        return points_new.detach()
    
    def expand_points_lines(self, points_a: th.Tensor, points_b: th.Tensor) -> th.Tensor:
        """Version of expand points that takes list of start and end points instead of connections.
        
        :param points_a: First points of the lines
        :param points_b: Second points of the lines
        """

        def average_points(point1, point2):
            points_new = (point1+point2)/2
            return points_new
        
        p1 = points_a
        p1 = p1.view([p1.shape[0], p1.shape[1], 2, -1])

        p5 = points_b
        p5 = p5.view([p5.shape[0], p5.shape[1], 2, -1])
        
        p3 = average_points(p1, p5)
        p2 = average_points(p1, p3)
        p4 = average_points(p3, p5)
        p1_5 = average_points(p1, p2)
        p2_5 = average_points(p2, p3)
        p3_5 = average_points(p3, p4)
        p4_5 = average_points(p4, p5)

        points_new = th.cat((p1.view_as(points_a), p1_5.view_as(points_a), p2.view_as(points_a),
            p2_5.view_as(points_a), p3.view_as(points_a), p3_5.view_as(points_a), p4.view_as(points_a), p4_5.view_as(points_a), p5.view_as(points_a)), 2)

        return points_new.detach()

    def create_image(self, points, connections, room_indices, img_size=256, res=200):
        raise NotImplementedError("This function seems to not be used")

    def forward(self, x, timesteps, xtalpha, epsalpha, is_syn=False, x_structural: th.Tensor=None, **kwargs) -> typing.Tuple[th.Tensor, th.Tensor]:
        """
        Apply the model to an input batch.

        :param x: an [N x S x C] Tensor of inputs (N x 2 x num corners) (2 because 2D coordinates).
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x S x C] Tensor of outputs.
        """

        assert x.shape[1] == 2, f"Expected x.shape[1] == 2, got {x.shape[1]}"
        
        if self.use_structural_cross_attention:
            # if x_structural is None:
            #     raise ValueError("Expected x_structural to be not None")

            if "struct_corners_a" not in kwargs:
                raise ValueError("Expected struct_corners_a in kwargs")
            
            if "struct_corners_b" not in kwargs:
                raise ValueError("Expected struct_corners_b in kwargs")

        # prefix = 'syn_' if is_syn else '' 

        prefix = ""

        # x is  i think batch_size x num_coordinates x number of corners
        x = x.permute([0, 2, 1]).float() # -> convert [N x C x S] to [N x S x C]
        # So now, x is batch_size x number of corners x num_coordinates (.i.e. N x corners x features)

        if not self.analog_bit:
            x = self.expand_points(x, kwargs[f'{prefix}connections'])

        if self.use_structural_cross_attention:
            # x_structural = x_structural.permute([0, 2, 1]).float()

            # if not self.analog_bit:
            #     x_structural = self.expand_points(x_structural, kwargs[f'{prefix}structural_connections'])

            struct_corners_a = kwargs["struct_corners_a"].permute([0, 2, 1]).float()
            struct_corners_b = kwargs["struct_corners_b"].permute([0, 2, 1]).float()

            if not self.analog_bit:
                x_structural = self.expand_points_lines(struct_corners_a, struct_corners_b)
        

        # Different input embeddings (Input, Time, Conditions) 
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        time_emb = time_emb.unsqueeze(1)
        
        # Embedding of the corners of shape: N x corners x feats (feats==self.model_channels)
        input_emb = self.input_emb(x)
        
        if self.condition_channels <= 1:
            raise ValueError("Condition channels must be > 1")
        
        cond: th.Tensor = None
        for key in [f'{prefix}room_types', f'{prefix}corner_indices', f'{prefix}room_indices']:
            if cond is None:
                cond = kwargs[key]
            else:
                cond = th.cat((cond, kwargs[key]), 2)
        
        assert cond is not None, "Condition is None, maybe the keys were wrong?"

        cond_emb = self.condition_emb(cond.float())

        # PositionalEncoding and DM model
        out = input_emb + cond_emb + time_emb.repeat((1, input_emb.shape[1], 1))

    
        if self.use_structural_cross_attention:

            door_type_emb = self.door_type_emb(th.eye(3).to(x.device).float())

            # TODO: mix in structural cond information? type and index are currently not useful, but corner_index might be?
            # Or maybe it's a bad idea to put in corner_index because of overfitting. Other than for the room corners,
            # the structural corners are not diffused, to they stay the same.
            struct_emb = self.struct_emb(x_structural)

            if self.use_wall_self_attention:
                wall_out = struct_emb
                
                for layer in self.wall_self_attention_layers:
                    wall_out = layer(wall_out, kwargs[f'{prefix}wall_self_mask'])

                struct_emb = struct_emb + wall_out

            for layer in self.transformer_layers:
                    out = layer(out, 
                                self_mask=kwargs[f'{prefix}self_mask'],
                                gen_mask=kwargs[f'{prefix}gen_mask'], 
                                x_structural=struct_emb, 
                                structural_mask=kwargs[f'{prefix}structural_mask'],
                                door_only_mask=kwargs[f'{prefix}door_only_mask'],
                                passage_ony_mask=kwargs[f'{prefix}passage_only_mask'],
                                entrance_only_mask=kwargs[f'{prefix}entrance_only_mask'],
                                door_type_emb=door_type_emb)
        else:
            for layer in self.transformer_layers:
                out = layer(out, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'])

        out_dec = self.output_linear1(out)
        out_dec = self.activation(out_dec)
        out_dec = self.output_linear2(out_dec)
        out_dec = self.output_linear3(out_dec)

        if not self.analog_bit:
            out_bin_start = x*xtalpha.repeat([1,1,9]) - out_dec.repeat([1,1,9]) * epsalpha.repeat([1,1,9])
            out_bin = (out_bin_start/2 + 0.5) # -> [0,1]
            out_bin = out_bin * self.img_size #-> [0, 256]
            out_bin = dec2bin(out_bin.round().int(), self.num_bits)
            out_bin_inp = out_bin.reshape([x.shape[0], x.shape[1], (2 * self.num_bits) * 9])
            out_bin_inp[out_bin_inp==0] = -1

            out_bin = th.cat((out_bin_start, out_bin_inp, cond_emb), 2)
            out_bin = self.activation(self.output_linear_bin1(out_bin))

            if self.use_structural_cross_attention:
                struct_emb = struct_emb
                
                struct_bin_input = (x_structural / 2) + 0.5 # -> [0,1]
                struct_bin_input = struct_bin_input * self.img_size #-> [0, img_size]
                struct_bin_input = dec2bin(struct_bin_input.round().int(), self.num_bits)
                struct_bin_input = struct_bin_input.reshape([x_structural.shape[0], x_structural.shape[1], (2 * self.num_bits) * 9])
                struct_bin_input[struct_bin_input==0] = -1

                struct_bin_embed = th.cat((x_structural, struct_bin_input, struct_emb), 2)
                struct_bin_embed = self.activation(self.struct_linear_bin1(struct_bin_embed))

                for layer in [self.output_linear_bin2, self.output_linear_bin3]:
                    out_bin = layer(out_bin, 
                                    self_mask=kwargs[f'{prefix}self_mask'], 
                                    gen_mask=kwargs[f'{prefix}gen_mask'],
                                    x_structural=struct_bin_embed,
                                    structural_mask=kwargs[f'{prefix}structural_mask'],
                                    door_only_mask=kwargs[f'{prefix}door_only_mask'],
                                    passage_ony_mask=kwargs[f'{prefix}passage_only_mask'],
                                    entrance_only_mask=kwargs[f'{prefix}entrance_only_mask'],
                                    door_type_emb=door_type_emb)
            else:
                out_bin = self.output_linear_bin2(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'])
                out_bin = self.output_linear_bin3(out_bin, kwargs[f'{prefix}door_mask'], kwargs[f'{prefix}self_mask'], kwargs[f'{prefix}gen_mask'])
            

            out_bin = self.output_linear_bin4(out_bin)

            out_bin = out_bin.permute([0, 2, 1]) # -> convert back [N x S x C] to [N x C x S]

        out_dec = out_dec.permute([0, 2, 1]) # -> convert back [N x S x C] to [N x C x S]

        if not self.analog_bit:
            return out_dec, out_bin
        else:
            return out_dec, None

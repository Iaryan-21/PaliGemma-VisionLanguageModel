from typing import Optional, Tuple 
import torch 
import torch.nn as nn 

class SigLipVisionConfig:
    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, num_channels=3, image_size=224, patch_size=16, attention_dropout=0.0, num_image_tokens: int = None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size # size of the embedding vector of vision transformer
        self.intermediate_size = intermediate_size # size of the linear layer that we use in the feed forward netwrok
        self.num_hidden_layers = num_hidden_layers # number layers of vision transfoemr 
        self.num_attention_heads = num_attention_heads # number of attention heads in multihead attention 
        self.num_channels = num_channels # number of channels in each image 
        self.patch_size = patch_size # size of each patch 
        self.image_size = image_size # size of image for PaliGemma needs to be 224, 448 or 896
        self.attention_dropout = attention_dropout 
        self.num_image_tokens = num_image_tokens # number of output token embeddings (how many image embeddings will be fo each image)


class SigLipVisionEmbeddings(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size 
        
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels,out_channels=self.embed_dim,kernel_size=self.patch_size,stride=self.patch_size,padding="valid")
        
        self.num_patches = (self.image_size//self.patch_size)**2
        self.num_positions = self.num_patches # positioanl encodings numbers 
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim) # These are learned because they can change accoridng to the need of the model.
        self.register_buffer(
            "position_ids",
            torch.arrange(self.num_positions).expand((1,-1)),
            persistent=False,
        ) 
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        
        # Convolve the patch size over the image with no overlapping.
        #output of convolution [Batch_size, Embed_dim, Num_Patches_H, Num_Patches_W]
        # Num_Patches_H // patch_size and  Num_Patches_W // patch_size
        
        patch_embeds = self.patch_embedding(pixel_values) # Num_Patches = Num_Patches_H + Num_patches_W For explanation
        embeddings = self.patch_embeds.flatten(2)
        
        # Chaning of dimensions [Batch_size, Embed_dim, Num_Patches] -> [Batch_size , Num_Patches, Embed_dim]

        embeddings = embeddings.transpose(1,2)
        
        # Addition of Positional Encodings to the Patch 
        
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        #[Batch_Size, Num_Patches, Embed_Dim]
        return embeddings
        
class SigLipEncoderLayer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.embed_dim = config.hidden_size
        self.self_attention = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual_1 = hidden_states
        hidden_states = self.layer_norm1(hidden_states) # Normalizing each of the dimension
        hidden_states, _ = self.self_attention(hidden_states = hidden_states)
        hidden_states = hidden_states + residual_1
        residual_2 = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states) # Transforms each of the layers independently
        hidden_states = hidden_states + residual_2
        
        return hidden_states
    
class SigLipEncoder(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = config
        self.layers = nn.ModuleList(
            [SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        
    def forward(self, input_embeds : torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
            
        return hidden_states

class SigLipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_state)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh") # GeLU instead of ReLU
        hidden_states = self.fc2(hidden_states)
        
        return hidden_states

class SigLipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.embed_dim = config.hidden_states 
        self.num_heads = config.num_attention_heads 
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # 1/squareroot(head_dim)
        self.dropout = config.attention_dropout
        
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim) # WK parametric matrices
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim) # WQ parametric matrices
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim) # WV parametric matrices
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim) # WO 
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len ,_ = hidden_states.size()
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        atten_weights = (torch.matmul(query,key.transpose(2,3))*self.scale)
        
        if atten_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            print(f"Value Error: {atten_weights.size()}")
        
        atten_weights = nn.functional.softmax(atten_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        atten_weights = nn.functional.dropout(atten_weights, p=self.dropout, training=self.training)
        atten_output = torch.matmul(atten_weights, value)
        
        if atten_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            print(f"Value Error: {atten_output.size()}")
        
        atten_output = atten_output.transpose(1,2).contiguous() # represent in a way in the memory such that the next operation does not require computation
        atten_output = atten_output.reshape(batch_size, seq_len, self.embed_dim) # concatenating the 1,2 dimensions
        atten_output = self.out_proj(atten_output)
        
        return atten_output, atten_weights
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = SigLipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values) # converting the image to bact of embedings
        last_hidden_state = self.encoder(input_embeds = hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state
    
    

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SigLipVisionConfig):
        self.config = config 
        self.vision_model = SiglipVisionTransformer(config)
        
    def forward(self, pixel_value) -> Tuple:
    #    [Batch Size , Channels, Height, Width] -> [Batch_size, Num Pathches, Embedded Dimension]
        return self.vision_model(pixel_values=pixel_value)
    
        

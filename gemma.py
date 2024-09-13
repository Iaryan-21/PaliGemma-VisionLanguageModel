import torch 
import torch.nn as nn 
from typing import Optional, Tuple, List 
from torch.nn import CrossEntropyLoss
from siglip import SigLipVisionConfig, SiglipVisionModel


class GemmaConfig():
    def __init__(self,
                 vocab_size, # Total number of tokens in vocabulary
                 hidden_size, #  size of embedding vector of each token 
                 intermediate_size, # size of the FeedForward Layer
                 num_hidden_layers, # Number of Layers of the Gemma Language Model
                 num_attention_heads, # Number of Heads for the Queries ( Grouped Query Attention)
                 num_key_value_heads, # Number of Heads for Key and Values 
                 head_dim = 256, # Number of dimesnion each head will work with.
                 max_position_embeddings = 8192, # maximum number of positions the model has been trained upon
                 rms_norm_eps = 1e-6, # parameter for rms normalization 
                 rope_theta = 10000.0, # rotary positional encoding theta , base frequency
                 attention_bias = False, 
                 attention_dropout = 0.0,
                 pad_token_id = None,
                 **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_state = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id



class PaliGemmaConfig():
    def __init__(self,
                 vision_config= None, # Configuration of Vision Encoder
                 text_config=None,    # Configuration of Text Encoder 
                 ignore_index = -100, 
                 image_token_index = 256000,
                 vocab_size = 257152,
                 projection_dim = 2048,
                 hidden_size = 2048,
                 pad_token_id = None,
                 **kwargs):

        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SigLipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) **2  # Number of Image tokens we get. 
        self.vision_config.projection_dim = projection_dim 

class KVCache():
    def __init__(self) -> None:
        self.key_cache : List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    '''
    Helper function that tells what the kv_cache contains. If it contains something then
    return the number of items it stores. The sequence length is the second last dimension, therefore 
    we return the -2. 
    Shape of key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
    '''
    def num_items(self)-> int: 
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Add the contents of key_states and value_states of this indexed layer.
    If neve anything has been added, we create this layer and if there is a token before
    we concatenate along the kv cache and value cache along [Batch_size, Num_heads_KV, Seq_Len, Head_Dim]
    '''
    if len(self.key_cache) <= layer_idx:
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
    else:
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

    return self.key_cache[layer_idx], self.value_cache[layer_idx]

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # eps is added to make the denominator non-zero
        self.weight = nn.Parameter(torch.zeros(dim)) # dimensions to be as same as the tokens 

    def norm(self,x):
        x = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return  x #root mean square statistic 1/rms(ai)
        
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0+self.weight.float()) # Gamma Learnable parameter multiplcation
        return output.type_as(x) 


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False) # Used by Activation function that Gemma model is using
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self,x):
        y = self.gate_proj(x) # [Batch_size, Seq_len, Hidden_size] -> [Batch_size, Seq_len, Intermediate_Size]
        y = torch.gelu(y,approximate="tanh") # [Batch_size, Seq_len, Intermediate_Size]
        j = self.up_proj(x) # [Batch_size, Seq_len, Hidden_Size] -> [Batch_size, Seq_len, Intermediate_Size]
        z = y * j # [Batch_size, Seq_len, Intermediate_Size]
        z = self.down_proj(x) # [Batch_size, Seq_len, Intermediate_Size] -> [Batch_size, Seq_len, Hidden_Size]
        return z

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads, = n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx:Optional[int] = None):
        super().__init__()
        self.config = config 
        self.layer_idx = layer_idx # Layer Index to know which KV Cache is to be used 

        self.attention_dropout = config.attention_dropout 
        self.hidden_size = config.hidden_size # size of embedding vector of each token
        self.num_heads = config.num_attention_heads 
        self.head_dim = config.head_dim # NUmber of dimensions each head will work with in MultiHeadAttention 
        self.num_key_value_heads = config.num_key_value_heads # heads for keys and values 
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # groups for keys and values 
        self.max_position_embeddings = config.max_position_embeddings # Positions that can be encoded in the positional encoding 
        self.rope_theta = config.rope_theta # Base Freq of ROPE 
        self.is_causal = True 

        assert self.hidden_size % self.num_heads == 0

        # For grouped query attention we have less heads for key and values which leads to smaller projection for the mebdding of each 
        # token when it is used as keys and values
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            kv_cache: Optional[KV_Cache] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(value_states, position_ids, Seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
        
        key_states = repeat_kv(key_states, self.num_key_value_groups) # Repeates the dimensions of key and valye which are not present for the current query heads. 
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim) # Attention

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, triaining = self.triaining)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output" is of the size " f"{attn_output.size()}"
                f"Needs to be: {(bsz, self.num_heads, q_len, self.head_dim)}"
            )
        
        attn_output = attn_output.transpose(1,2).contiguous() # The seq len should be the second dimension
        attn_output = attn_output.view(bsz, qlen, -1) # Concatenating all heads togethe.
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config:GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            kv_cache = kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
        

'''
Language Model : GEMMA 
'''
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config 
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None
        ) -> torch.FloatTensor:
        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states,attention_mask = attention_mask,position_ids = position_ids,kv_cache = kv_cache)
        
        hidden_states = self.norm(hidden_states)

        return hidden_states

'''
TRANSFORMER MODEL LM
In hugging face, it is generally transformer model plus causal model.
'''
class GemmaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # Linear layer in the transformer that porjects each head into logits.

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight 

    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None
        ) -> Tuple:

        outputs = self.model(attention_mask = attention_mask, position_ids = position_ids, input_embeds=inputs_embeds, kv_cache=kv_cache)
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits" : logits
        }

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
    
class PaliGemmaMultimodalProjector(nn.Module):
    '''
    Linear Layer that converts the size of the Image features extraced from the vision encoder into the same size of the embedding size 
    that is used by the language model.
    '''
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states

# It is called conditional generation as we are conditioning the generation of text on the image input.
class PaliGemmaConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config 
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultimodalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tieweights()
    
    def _merge_input_ids_with_image_features(self, image_features: torch.Tensor, input_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KV_Cache] = None):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = input_embeds.dtype, input_embeds.device 
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device) # This is going to hold the combined tensor.
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id) # To recognize a text token which is basically not an image token and not a padding token 
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expanding the mask to create the batch size dimension and the sequence dimension

        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding) # if text_mask_explained is True it will copy in input_embeds else final_embedding
        '''
         Cannot use torch.where because the scaled_image_features length is not equal to the sequence length of final embedding
        '''
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)  # copy from scaled_image_features where image_mask_explained is true
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding) # wherever the pad_masked_expaneded is true keep them as zero embeddings or keep the final embeddings as they are 

        dtype , device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        '''
        This only works when we are in the prefill phase.
        It only works when there is no padding
        '''
        if kv_cache is None or kv_cache.num_items() == 0:
            causal_mask = torch.full((batch_size, q_len, q_len), fill_volume = 0, dtype=dtype, device=device)
        else:
            assert q_len == 1 # Generating new tokens, query must be one single token
            kv_len = kv_cache.num_items() + q_len
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value = 0, dtype=dtype, device=device # This is not applied on text prompt because we need to have an indication of the total text and what is its purpose
            )
        causal_mask = causal_mask.unsqueeze(1)
        '''
        Image token does not need to be causal as each text token generated should be able to work on all the images.
        Textual prompt is not causal as well, the text prompt is usally very short and we want to describe what the task is and hence masking it would not be ideal.
        Suffix / Target Query needs to be Causal . It needs to bae able to consider all the tokens in sequnce of images and texts and then the last generated output token.
        Causal only on generated text. 
        In this way, the information about the prompt is replicated in each of this token. Included information about future tokens that are part of the prompt.
        When working with KVCache during inference we don't have a causal mask
        '''
        # Positions of  tokens used by the Rotatory Position Encodings

        if kv_cache is not None and kv_cache.num_items() > 0 : # N tokens that are part of the prompt of the user
            position_ids = attention_mask.cumsum(-1)[:,-1] # Positions to apply the rotary position encoding, need upto total number of tokens in the prompt. 
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0) # 
            else:
                position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)  # This will generate one single mask which is the position corresponding to the last token. 
               
        return final_embedding, causal_mask, position_ids

    def forward(self, input_ids: torch.LongTensor = None, pixel_values: torch.FloatTensor = None, attention_mask: Optional[torch.Tensor]= None, kv_cache: Optional[KVCache] = None) -> Tuple:
        '''
        input_ids: Output of Paligemmaprocessor, lot of image tokens, Bos, Prompt of user, seperator, EOS
        pixel_values: image extracted from PaliGemma processor , rescaled, resized and normalized
        attention_mask: comes directly from the tokenizer
        kv_cache: XX 
        Since no padding has been used, the attention_mask output will be a series of 1's. 
        '''
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        input_embeds = self.language_model.get_input_embeddings()(input_ids) # Extra the input embeddings, shape: (Batch_size, Seq_Len, Hidden_states)
        select_image_features = self.vision_tower(pixel_values.to(input_embeds.dtype)) # Merge Text and Images : [Batch_size, Num_patches, Embed_Dim]
        image_features = self.multi_modal_projector(select_image_features) # [Batch_size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_State]
        input_embeds, attention_mask , position_ids = self._merge_input_ids_with_image_features(image_features, input_embeds, input_ids, attention_mask, kv_cache) # Merging embeddings of text and image tokens 

        outputs = self.language_model(
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_embeds = input_embeds,
            kv_cache = kv_cache
        )

        return outputs 
    


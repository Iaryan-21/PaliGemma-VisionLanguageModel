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

        
    def forward(self, input_ids: torch.LongTensor = None, pixel_values: torch.FloatTensor = None, attention_mask: Optional[torch.Tensor]= None, kv_cache: Optional[KVCache] = None) -> Tuple:
        # input_ids: Output of Paligemmaprocessor, lot of image tokens, Bos, Prompt of user, seperator, EOS
        # pixel_values: image extracted from PaliGemma processor , rescaled, resized and normalized
        # attention_mask: comes directly from the tokenizer
        # kv_cache: XX 
        # Since no padding has been used, the attention_mask output will be a series of 1's. 
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
    


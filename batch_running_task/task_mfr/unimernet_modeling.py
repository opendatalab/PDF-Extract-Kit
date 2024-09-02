import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
from ftfy import fix_text
from transformers import PreTrainedTokenizerFast
from transformers import DonutSwinConfig, VisionEncoderDecoderConfig
from transformers import AutoModel, VisionEncoderDecoderModel, AutoImageProcessor, MBartForCausalLM
from unimernet_processor import VariableDonutProcessor, VariableDonutImageProcessor
from transformers.models.mbart.modeling_mbart import MBartDecoder
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput, CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.donut.modeling_donut_swin import DonutSwinPatchEmbeddings, DonutSwinEmbeddings, DonutSwinModel, DonutSwinEncoder
from transformers.utils import logging, ModelOutput


logger = logging.get_logger(__name__)


class VariableDonutSwinConfig(DonutSwinConfig):
    pass


class VariableDonutSwinEmbeddings(DonutSwinEmbeddings):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__(config, use_mask_token)

        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None
        self.position_embeddings = None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))

        self.row_embeddings = None
        self.column_embeddings = None
        if config.use_2d_embeddings:
            self.row_embeddings = nn.Parameter(torch.zeros(1, self.patch_grid[0] + 1, config.embed_dim))
            self.column_embeddings = nn.Parameter(torch.zeros(1, self.patch_grid[1] + 1, config.embed_dim))

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:

        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # Layernorm across the last dimension (each patch is a single row)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, embed_dim = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings[:, :seq_len, :]

        if self.row_embeddings is not None and self.column_embeddings is not None:
            # Repeat the x position embeddings across the y axis like 0, 1, 2, 3, 0, 1, 2, 3, ...
            row_embeddings = self.row_embeddings[:, :output_dimensions[0], :].repeat_interleave(output_dimensions[1],
                                                                                                dim=1)
            column_embeddings = self.column_embeddings[:, :output_dimensions[1], :].repeat(1, output_dimensions[0], 1)

            embeddings = embeddings + row_embeddings + column_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions

class VariableDonutSwinModel(DonutSwinModel):
    config_class = VariableDonutSwinConfig

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = VariableDonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


@dataclass
class CausalLMOutputWithCrossAttentionsAndCounting(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    counting: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomMBartDecoder(MBartDecoder):
    def __init__(self, config):
        print("CustomMBartDecoder init")
        super().__init__(config)
        hidden_size = config.d_model
        self.counting_context_weight = nn.Sequential(
            nn.Linear(config.vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, config.d_model)
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        count_pred: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_shape = input.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            else:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)

        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)

        # TODO: add counting context weight to hidden_states
        if count_pred is not None:
            count_context_weight = self.counting_context_weight(count_pred)
            hidden_states = hidden_states + 0.5 * count_context_weight.unsqueeze(1)
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class CustomMBartForCausalLM(MBartForCausalLM):
    def __init__(self, config, length_aware=True):
        print("CustomMBartForCausalLM init")
        super().__init__(config)
        # Modify the decoder within MBartDecoderWrapper
        self.model.decoder = CustomMBartDecoder(config)
        self.counting_decoder = SeqCountingDecoder(config.d_model, config.vocab_size)
        self.length_aware = length_aware
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        count_gt: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                if the model is configured as a decoder.
            encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MBartForCausalLM

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
        >>> model = MBartForCausalLM.from_pretrained("facebook/mbart-large-cc25", add_cross_attention=False)
        >>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
        >>> list(logits.shape) == expected_shape
        True
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.length_aware:
            count_pred = self.counting_decoder(encoder_hidden_states)
        else:
            count_pred = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            count_pred=count_pred,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits, count_pred) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentionsAndCounting(
            loss=loss,
            logits=logits,
            counting=count_pred,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, config, length_aware=True):
        print("CustomVisionEncoderDecoderModel init")
        super().__init__(config)
        # Replace the MBartForCausalLM with your CustomMBartForCausalLM
        self.decoder = CustomMBartForCausalLM(self.config.decoder, length_aware=length_aware)
        self.length_aware = length_aware

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, VisionEncoderDecoderModel
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        >>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        >>> # load image from the IAM dataset
        >>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

        >>> # training
        >>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        >>> model.config.pad_token_id = processor.tokenizer.pad_token_id
        >>> model.config.vocab_size = model.config.decoder.vocab_size

        >>> pixel_values = processor(image, return_tensors="pt").pixel_values
        >>> text = "hello world"
        >>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
        >>> outputs = model(pixel_values=pixel_values, labels=labels)
        >>> loss = outputs.loss

        >>> # inference (generation)
        >>> generated_ids = model.generate(pixel_values)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))
            count_gt = kwargs_decoder.get("count_gt", None)
            if self.length_aware and count_gt is not None:
                count_gt = torch.log(count_gt.float() + 1)
                count_pred = decoder_outputs.counting if return_dict else decoder_outputs[1] # dtype: torch.float32
                counting_loss_fct = nn.SmoothL1Loss()
                counting_loss = counting_loss_fct(count_pred, count_gt)
                # print("counting_loss", counting_loss)
                loss += counting_loss


        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    

class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm(attn_output + x)
        return x

class SeqCountingDecoder(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, num_layers=4):
        super(SeqCountingDecoder, self).__init__()
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(embed_size=in_features, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc2 = nn.Linear(in_features // 2, out_features)

    def forward(self, x):
        # x shape (bs, seq_len, d_model)
        for block in self.attention_blocks:
            x = block(x)
        x = self.fc1(x)  # Process each patch embedding
        x = self.relu(x)
        x = x.transpose(1, 2)  # Prepare for global pooling
        x = self.global_avg_pool(x)  # Global pooling
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.fc2(x)  # Predict token counts
        return x # (bs, vocab_size)


class DonutEncoderDecoder(nn.Module):

    def __init__(self, model_name, num_tokens, pad_token_id, bos_token_id, eos_token_id, length_aware=True):
        super().__init__()
        config = VisionEncoderDecoderConfig.from_pretrained(model_name)
        # encoder_config = vars(config.encoder)
        # encoder = DonutSwinConfig(**encoder_config)
        # config.encoder = encoder
        # self.config = config

        # AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)
        # # provider_options = {
        # #     "trt_engine_cache_enable": True,
        # #     "trt_engine_cache_path": "weights/unimernet_tensorRT/rt_cache"
        # # }

        # # self.model = ORTModelForVision2Seq.from_pretrained("weights/unimernet_tensorRT", config=config, provider="TensorrtExecutionProvider", provider_options=provider_options)

        # self.model = CustomVisionEncoderDecoderModel.from_pretrained(model_name, config=self.config, length_aware=length_aware)
        self.model = VisionEncoderDecoderModel.from_pretrained("weights/unimernet_clean")

        self.model.config.decoder_start_token_id = bos_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.eos_token_id = eos_token_id
        self.model.decoder.resize_token_embeddings(num_tokens)
        self.pad_token_id = pad_token_id

    def forward(self, pixel_values, decoder_input_ids, decoder_attention_mask, **kwargs):
        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        labels = decoder_input_ids * 1
        labels = labels.masked_fill(labels == self.pad_token_id, -100)

        loss = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids[:, :-1],
            decoder_attention_mask=decoder_attention_mask[:, :-1],
            labels=labels[:, 1:],
            **kwargs # for Length-Aware Module
        ).loss
        return loss

    @torch.no_grad()
    def generate(self, pixel_values, temperature, max_new_tokens, decoder_start_token_id, do_sample, top_p,
                 **kwargs):

        num_channels = pixel_values.shape[1]
        if num_channels == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        outputs = self.model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            decoder_start_token_id=decoder_start_token_id,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
        )
        return outputs[:, 1:]



class DonutTokenizer:
    def __init__(self, path):
        AutoImageProcessor.register(VariableDonutSwinConfig, VariableDonutImageProcessor)
        processor = VariableDonutProcessor.from_pretrained(path)
        processor.train = False
        self.tokenizer = processor.tokenizer
        self.max_seq_len = 2048
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, texts, max_length=None):
        if not max_length:
            max_length = self.max_seq_len
        text_inputs = self.tokenizer(
            texts,
            return_token_type_ids=False,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        return text_inputs

    @staticmethod
    def post_process(text):
        text = fix_text(text)
        return text

    def token2str(self, tokens) -> list:
        generated_text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def detokenize(self, tokens):
        toks = [self.tokenizer.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ''
                toks[b][i] = toks[b][i].replace('Ġ', ' ').strip()
                if toks[b][i] in ([self.tokenizer.bos_token, self.tokenizer.eos_token, self.tokenizer.pad_token]):
                    del toks[b][i]
        return toks

AutoImageProcessor.register(VariableDonutSwinConfig, VariableDonutImageProcessor)
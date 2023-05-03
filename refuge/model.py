# Copyright (C) Alex Carpenter and Simon Biggs

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import cast

import torch
from torch import nn
from torch.nn.functional import cosine_similarity
from transformers import GPTNeoXForCausalLM


class GPTNeoXPromptTuningLM(GPTNeoXForCausalLM):
    def __init__(self, config):
        self._soft_prompt_parameter = None
        self._step = None
        self._soft_tokens = None

        super().__init__(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = cast(
            GPTNeoXPromptTuningLM,
            super().from_pretrained(pretrained_model_name_or_path, **kwargs),
        )

        for param in model.parameters():
            param.requires_grad = False

        return model

    @property
    def soft_prompt_parameter(self):
        assert self._soft_prompt_parameter is not None
        return self._soft_prompt_parameter

    @property
    def soft_prompt_embeddings(self):
        return self.soft_prompt_parameter.data

    @soft_prompt_embeddings.setter
    def soft_prompt_embeddings(self, embeddings: torch.Tensor):
        self._soft_prompt_parameter = nn.parameter.Parameter(embeddings.to(self.device))

    @property
    def step(self):
        assert self._step is not None
        return self._step

    @step.setter
    def step(self, step: int):
        self._step = step

    @property
    def soft_tokens(self):
        assert self._soft_tokens is not None
        return self._soft_tokens

    @soft_tokens.setter
    def soft_tokens(self, soft_tokens: list[str]):
        self._soft_tokens = soft_tokens

    @property
    def soft_prompt(self):
        return "".join(self.soft_tokens)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)

    def convert_tokens_to_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        soft_token_mask = tokens >= self.config.vocab_size
        masked_tokens = tokens.clone()
        masked_tokens[soft_token_mask] = 0

        embedding_layer = cast(nn.Embedding, self.gpt_neox.embed_in)
        embedding = embedding_layer(masked_tokens)

        soft_token_idx = tokens[soft_token_mask] - self.config.vocab_size

        soft_embedding = self.soft_prompt_embeddings[soft_token_idx]
        embedding[soft_token_mask] = soft_embedding

        return embedding

    def convert_logits_to_tokens(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            input_shape = input_ids.shape
            attention_mask = input_ids.new_ones(input_shape)

        model_inputs: dict[str, torch.Tensor | None] = {}

        try:
            inputs_embeds = self.convert_tokens_to_embeddings(input_ids)
            model_inputs = {"inputs_embeds": inputs_embeds}
        except AssertionError:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
        )

        return model_inputs

    def convert_embeddings_to_token(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings_transposed = embeddings.T
        normalized_embedding_transposed = torch.linalg.norm(
            embeddings_transposed, dim=0, keepdim=True
        )

        embedding_layer = cast(nn.Embedding, self.gpt_neox.embed_in)
        normalized_weights = torch.linalg.norm(
            embedding_layer.weight, dim=1, keepdim=True
        )

        cosine_similarities = (
            (embedding_layer.weight @ embeddings_transposed)
            / (normalized_weights @ normalized_embedding_transposed)
        ).T

        token_indices = torch.argmax(cosine_similarities, dim=1)

        return token_indices

    def translated_soft_prompt(self):
        return self.convert_embeddings_to_token(self.soft_prompt_embeddings)

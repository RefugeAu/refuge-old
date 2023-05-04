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


from typing import Optional, cast

import torch
from torch import nn
from transformers import GPTNeoXForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class GPTNeoXPromptTuningLM(GPTNeoXForCausalLM):
    def __init__(self, config):
        self._soft_prompt_parameter = None
        self._step = None
        self._soft_tokens = None
        self.run_soft_prompt_loss = False

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
        **kwargs,
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

    def cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings_transposed = embeddings.T
        embedding_normalisation_factor = torch.linalg.norm(
            embeddings_transposed, dim=0, keepdim=True
        )

        embedding_layer = cast(nn.Embedding, self.gpt_neox.embed_in)
        weights_normalisation_factor = torch.linalg.norm(
            embedding_layer.weight, dim=1, keepdim=True
        )

        cosine_similarities = (
            (embedding_layer.weight @ embeddings_transposed)
            / (weights_normalisation_factor @ embedding_normalisation_factor)
        ).T

        return cosine_similarities

    def convert_embeddings_to_token(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine_similarities = self.cosine_similarity(embeddings)

        token_indices = torch.argmax(cosine_similarities, dim=1)

        return token_indices

    def translated_soft_prompt(self):
        return self.convert_embeddings_to_token(self.soft_prompt_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        results = cast(
            CausalLMOutputWithPast,
            super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            ),
        )

        if results.loss is None or not self.run_soft_prompt_loss:
            return results

        cosine_similarities = self.cosine_similarity(self.soft_prompt_parameter)

        max_token_results = torch.max(cosine_similarities, dim=1)
        max_token_affinity = cast(torch.Tensor, max_token_results.values - 1e-5).to(
            self.device
        )
        target_max_token_affinity = torch.ones_like(max_token_affinity)

        token_affinity_loss_function = nn.BCELoss()
        token_affinity_loss: torch.Tensor = token_affinity_loss_function(
            max_token_affinity, target_max_token_affinity
        )

        # soft_prompt_token_ids = max_token_results.indices.unsqueeze(0)

        # soft_prompt_results = cast(
        #     CausalLMOutputWithPast,
        #     super().forward(
        #         input_ids=soft_prompt_token_ids, labels=soft_prompt_token_ids
        #     ),
        # )

        # one_hot = nn.functional.one_hot(
        #     soft_prompt_token_ids[0, ...], num_classes=50280
        # ).float()
        # selected = torch.diag(one_hot @ soft_prompt_results.logits[0, ...].T)
        # likelihood_of_prompt = nn.functional.sigmoid(selected)
        # target_likelihood = torch.ones_like(likelihood_of_prompt)

        # prompt_likelihood_loss_function = nn.BCELoss()
        # prompt_likelihood_loss: torch.Tensor = 100 * prompt_likelihood_loss_function(
        #     likelihood_of_prompt, target_likelihood
        # )

        combined_loss = results.loss + token_affinity_loss  # + prompt_likelihood_loss

        # print(
        #     f"Original loss: {results.loss} | Token affinity loss: {token_affinity_loss} | Prompt likelihood loss: {prompt_likelihood_loss}"
        # )

        results.loss = cast(torch.FloatTensor, combined_loss)

        return results

    # def _patched_forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[tuple[tuple[torch.FloatTensor]]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> tuple | CausalLMOutputWithPast:
    #     return_dict = (
    #         return_dict if return_dict is not None else self.config.use_return_dict
    #     )

    #     outputs = self.gpt_neox(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         past_key_values=past_key_values,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     hidden_states = outputs[0]
    #     lm_logits = self.embed_out(hidden_states)

    #     lm_loss = None
    #     if labels is not None:
    #         # move labels to correct device to enable model parallelism
    #         labels = labels.to(lm_logits.device)
    #         # we are doing next-token prediction; shift prediction scores and input ids by one
    #         shift_logits = lm_logits[:, :-1, :].contiguous()
    #         labels = labels[:, 1:].contiguous()
    #         predicted = shift_logits.view(-1, shift_logits.size(-1))
    #         target = labels.view(-1)

    #         # TODO: Improve this custom value of 100
    #         weight = torch.ones_like(target).float()
    #         weight[0 : len(self._soft_tokens) - 1] = 20

    #         loss_fct = nn.modules.loss.CrossEntropyLoss(reduction="none")
    #         per_token_loss = loss_fct(predicted, target)

    #         lm_loss = torch.mean(per_token_loss * weight)

    #     if not return_dict:
    #         output = (lm_logits,) + outputs[1:]
    #         return ((lm_loss,) + output) if lm_loss is not None else output

    #     return CausalLMOutputWithPast(
    #         loss=lm_loss,
    #         logits=lm_logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )

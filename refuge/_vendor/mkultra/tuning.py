# MIT License

# Copyright (c) 2021 corolla-johnson

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable = unused-argument

from typing import cast

import torch
from torch import nn
from transformers import GPTNeoXForCausalLM


class GPTNeoXPromptTuningLM(GPTNeoXForCausalLM):
    def __init__(self, config):
        self._soft_prompt_parameter = None

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

    def prepare_inputs_for_generation(
        self, input_ids, *args, past_key_values=None, **kwargs
    ):
        input_ids = input_ids.to(self.device)
        # Drop 'past' to make things easier for us later
        return super().prepare_inputs_for_generation(input_ids, None, *args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        kwargs["input_ids"] = kwargs["input_ids"].to(self.device)

        return super().generate(*args, **kwargs)

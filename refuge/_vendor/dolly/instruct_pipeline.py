import logging
import re
from typing import List, TypedDict, cast

import torch
from transformers import Pipeline, PreTrainedTokenizer

logger = logging.getLogger(__name__)

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

PROMPT_FOR_GENERATION_FORMAT = f"""{INTRO_BLURB}

{INSTRUCTION_KEY}
{"{instruction}"}

{RESPONSE_KEY}
"""


class ModelOutput(TypedDict):
    generated_sequence: torch.Tensor
    input_ids: torch.Tensor
    instruction_text: torch.Tensor


def get_special_token_id(tokenizer: PreTrainedTokenizer, key: str) -> int:
    """Gets the token ID for a given string that has been added to the tokenizer as a special token.

    When training, we configure the tokenizer so that the sequences like "### Instruction:" and "### End" are
    treated specially and converted to a single, new token.  This retrieves the token ID each of these keys map to.

    Args:
        tokenizer (PreTrainedTokenizer): the tokenizer
        key (str): the key to convert to a single token

    Raises:
        RuntimeError: if more than one ID was generated

    Returns:
        int: the token ID for the given key
    """
    token_ids = tokenizer.encode(key)
    if len(token_ids) > 1:
        raise ValueError(
            f"Expected only a single token for '{key}' but found {token_ids}"
        )
    return token_ids[0]


class InstructionTextGenerationPipeline(Pipeline):
    def __init__(
        self,
        *args,
        do_sample: bool = True,
        max_new_tokens: int = 256,
        top_p: float = 0.92,
        top_k: int = 0,
        **kwargs,
    ):
        """Initialize the pipeline

        Args:
            do_sample (bool, optional): Whether or not to use sampling. Defaults to True.
            max_new_tokens (int, optional): Max new tokens after the prompt to generate. Defaults to 128.
            top_p (float, optional): If set to float < 1, only the smallest set of most probable tokens with
                probabilities that add up to top_p or higher are kept for generation. Defaults to 0.92.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
                Defaults to 0.
        """
        super().__init__(
            *args,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def _sanitize_parameters(
        self, return_full_text: bool | None = None, **generate_kwargs
    ):
        preprocess_params = {}

        assert self.tokenizer is not None

        # newer versions of the tokenizer configure the response key as a special token.  newer versions still may
        # append a newline to yield a single token.  find whatever token is configured for the response key.
        tokenizer_response_key = next(
            (
                token
                for token in self.tokenizer.additional_special_tokens
                if token.startswith(RESPONSE_KEY)
            ),
            None,
        )

        response_key_token_id = None
        end_key_token_id = None
        if tokenizer_response_key:
            try:
                response_key_token_id = get_special_token_id(
                    self.tokenizer, tokenizer_response_key
                )
                end_key_token_id = get_special_token_id(self.tokenizer, END_KEY)

                # Ensure generation stops once it generates "### End"
                generate_kwargs["eos_token_id"] = end_key_token_id
            except ValueError:
                pass

        forward_params = generate_kwargs
        postprocess_params = {
            "response_key_token_id": response_key_token_id,
            "end_key_token_id": end_key_token_id,
        }

        if return_full_text is not None:
            postprocess_params["return_full_text"] = return_full_text

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, input_, **_kwargs):
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction=input_)

        assert self.tokenizer is not None

        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        inputs["prompt_text"] = prompt_text
        inputs["instruction_text"] = input_
        return inputs

    def _forward(
        self, input_tensors: dict[str, torch.Tensor], **generate_kwargs
    ) -> ModelOutput:
        input_ids = input_tensors["input_ids"]
        attention_mask = input_tensors.get("attention_mask", None)

        if input_ids.shape[1] == 0:
            input_ids = None
            attention_mask = None
            in_b = 1
        else:
            in_b = input_ids.shape[0]

        assert input_ids is not None
        assert self.tokenizer is not None

        generated_sequence = cast(
            torch.LongTensor,
            self.model.generate(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device)
                if attention_mask is not None
                else None,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            ),
        )

        out_b = generated_sequence.shape[0]

        generated_sequence = generated_sequence.reshape(
            in_b, out_b // in_b, *generated_sequence.shape[1:]
        )

        instruction_text = input_tensors.pop("instruction_text")
        return {
            "generated_sequence": generated_sequence,
            "input_ids": input_ids,
            "instruction_text": instruction_text,
        }

    def postprocess(
        self,
        model_outputs: ModelOutput,
        response_key_token_id,
        end_key_token_id,
        return_full_text: bool = False,
    ):
        generated_sequence_tensor = model_outputs["generated_sequence"][0]
        instruction_text = model_outputs["instruction_text"]

        generated_sequence: List[List[int]] = generated_sequence_tensor.numpy().tolist()
        records = []
        for sequence in generated_sequence:
            # The response will be set to this variable if we can identify it.
            decoded = None

            # If we have token IDs for the response and end, then we can find the tokens and only decode between them.
            if response_key_token_id and end_key_token_id:
                # Find where "### Response:" is first found in the generated tokens.  Considering this is part of the
                # prompt, we should definitely find it.  We will return the tokens found after this token.
                try:
                    response_pos = sequence.index(response_key_token_id)
                except ValueError:
                    logger.warn(
                        f"Could not find response key {response_key_token_id} in: {sequence}"
                    )
                    response_pos = None

                if response_pos:
                    # Next find where "### End" is located.  The model has been trained to end its responses with this
                    # sequence (or actually, the token ID it maps to, since it is a special token).  We may not find
                    # this token, as the response could be truncated.  If we don't find it then just return everything
                    # to the end.  Note that even though we set eos_token_id, we still see the this token at the end.
                    try:
                        end_pos = sequence.index(end_key_token_id)
                    except ValueError:
                        end_pos = None

                    assert self.tokenizer is not None
                    decoded = self.tokenizer.decode(
                        sequence[response_pos + 1 : end_pos]
                    ).strip()

            if not decoded:
                # Otherwise we'll decode everything and use a regex to find the response and end.
                assert self.tokenizer is not None
                fully_decoded = self.tokenizer.decode(sequence)

                # The response appears after "### Response:".  The model has been trained to append "### End" at the
                # end.
                m = re.search(
                    r"#+\s*Response:\s*(.+?)#+\s*End", fully_decoded, flags=re.DOTALL
                )

                if m:
                    decoded = m.group(1).strip()
                else:
                    # The model might not generate the "### End" sequence before reaching the max tokens.  In this case,
                    # return everything after "### Response:".
                    m = re.search(
                        r"#+\s*Response:\s*(.+)", fully_decoded, flags=re.DOTALL
                    )
                    if m:
                        decoded = m.group(1).strip()
                    else:
                        logger.warn(f"Failed to find response in:\n{fully_decoded}")

            # If the full text is requested, then append the decoded text to the original instruction.
            # This technically isn't the full text, as we format the instruction in the prompt the model has been
            # trained on, but to the client it will appear to be the full text.
            if return_full_text:
                decoded = f"{instruction_text}\n{decoded}"

            rec = {"generated_text": decoded}

            records.append(rec)

        return records

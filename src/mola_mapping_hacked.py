# coding=utf-8
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from peft.config import PeftConfig
from peft.mixed_model import PeftMixedModel
##newmodified
from peft.peft_model import (
    PeftModel,
    # PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
##newmodified
##as other PeftModels are not modified, only import the modified one
from src.mola_peft_model_hacked import  PeftModelForCausalLM

from peft.tuners import (
    AdaLoraConfig,
    AdaLoraModel,
    AdaptionPromptConfig,
    BOFTConfig,
    BOFTModel,
    IA3Config,
    IA3Model,
    LNTuningConfig,
    LNTuningModel,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    LoraConfig,
    #import modified LoraModel
    # LoraModel,
    MultitaskPromptTuningConfig,
    OFTConfig,
    OFTModel,
    PolyConfig,
    PolyModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    VeraConfig,
    VeraModel,
)
##newmodified
##LoraModel are needed to import for new version of mapping.py
from src.mola_lora_hacked import LoraModel

from peft.tuners.tuners_utils import BaseTuner as _BaseTuner
from peft.utils import _prepare_prompt_learning_config


if TYPE_CHECKING:
    from transformers import PreTrainedModel


MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[PeftModel]] = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM,
    "TOKEN_CLS": PeftModelForTokenClassification,
    "QUESTION_ANS": PeftModelForQuestionAnswering,
    "FEATURE_EXTRACTION": PeftModelForFeatureExtraction,
}

PEFT_TYPE_TO_CONFIG_MAPPING: dict[str, type[PeftConfig]] = {
    "ADAPTION_PROMPT": AdaptionPromptConfig,
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
    "LOHA": LoHaConfig,
    "LOKR": LoKrConfig,
    "ADALORA": AdaLoraConfig,
    "BOFT": BOFTConfig,
    "IA3": IA3Config,
    "MULTITASK_PROMPT_TUNING": MultitaskPromptTuningConfig,
    "OFT": OFTConfig,
    "POLY": PolyConfig,
    "LN_TUNING": LNTuningConfig,
    "VERA": VeraConfig,
}

PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[_BaseTuner]] = {
    "LORA": LoraModel,
    "LOHA": LoHaModel,
    "LOKR": LoKrModel,
    "ADALORA": AdaLoraModel,
    "BOFT": BOFTModel,
    "IA3": IA3Model,
    "OFT": OFTModel,
    "POLY": PolyModel,
    "LN_TUNING": LNTuningModel,
    "VERA": VeraModel,
}


def get_peft_config(config_dict: dict[str, Any]) -> PeftConfig:
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)

##newmodified
## add parameter number_experts and top_k for MoLA
def get_peft_model(
    model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default", mixed: bool = False, number_experts: list = [8] * 32, top_k: list = [2] * 32
) -> PeftModel | PeftMixedModel:
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]):
            Model to be wrapped.
        peft_config ([`PeftConfig`]):
            Configuration object containing the parameters of the Peft model.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        mixed (`bool`, `optional`, defaults to `False`):
            Whether to allow mixing different (compatible) adapter types.
    """
    model_config = getattr(model, "config", {"model_type": "custom"})
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()

    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    if mixed:
        return PeftMixedModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys() and not peft_config.is_prompt_learning:
        return PeftModel(model, peft_config, adapter_name=adapter_name)

    if peft_config.is_prompt_learning:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)
    ##newmodified
    ##call PeftModelForCausalLM
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config, adapter_name=adapter_name, number_experts=number_experts, top_k=top_k)


def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default"
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name)

    return peft_model.model


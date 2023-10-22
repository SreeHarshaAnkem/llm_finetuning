import sys
import torch

import numpy as np
import pandas as pd

import os
import random
import warnings

import ast
from loguru import logger


from collections import OrderedDict

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import logging


from peft import LoraConfig, get_peft_model
from peft import PromptEncoderConfig
from peft import PromptTuningConfig, PromptTuningInit
from peft import IA3Config

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore")


NUM_CLIENTS = 2
NUM_ROUNDS = 3


def get_peft_model_frm_config(net, peft_method, tokenizer=None):
    if peft_method == "lora":
        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
        )
    elif peft_method == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128
        )
    elif peft_method == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=8)
    elif peft_method == "ia3":
        peft_config = IA3Config(
            task_type="SEQ_CLS", target_modules=["query", "key", "value", "dense"]
        )
    net = get_peft_model(net, peft_config)
    return net


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_model(model_id, use_peft, peft_method, num_labels, id2label, label2id):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )  # "yikuan8/Clinical-BigBird")
    net = AutoModelForSequenceClassification.from_pretrained(
        model_id,  # "yikuan8/Clinical-BigBird",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)
    if use_peft:
        net = get_peft_model_frm_config(net, peft_method, tokenizer)
        print_trainable_parameters(net)
    return net

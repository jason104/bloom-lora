import gc
import os
import sys
import threading

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
    Trainer, TrainingArguments,
    BloomForCausalLM, BloomTokenizerFast,
)

from peft import LoraConfig, TaskType, get_peft_model
import transformers

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'all': 'bias',
}

def _perform_training_preparations(model, fine_tune_type, trainable_components, verbose=True):
    if fine_tune_type == 'frozen':
        trainable_components = []

    if fine_tune_type == 'full_ft':
        training_preparation(model,
                             encoder_trainable=True,
                             verbose=args.verbose)
    elif fine_tune_type in {'bitfit', 'frozen'}:
        training_preparation(model,
                             encoder_trainable=False,
                             trainable_components=trainable_components,
                             verbose=verbose)


def training_preparation(model, encoder_trainable, trainable_components=None,
                             verbose=True):
        """Performs training preparation.

        Perform training preparation including: model initialization, optimizer initialization, relevant
        gradients deactivation and plotting a list of all trainable params (if verbose is True).

        Args:
            learning_rate (float): learning_rate to train with.
            optimizer(str): optimizer to perform the training with, currently adam and adamw are supported.
            encoder_trainable (bool): if True will perform a Full-FT else will perform BitFit training preparation.
            trainable_components(Union[List[str], None]): list of trainable component.(subset of `BIAS_TERMS_DICT` keys)
            verbose: if True will plot a list of all trainable params

        """

        if encoder_trainable and trainable_components:
            raise Exception(
                f"If encoder_trainable is True, you shouldn't supply trainable_components. "
                f"Got trainable_components: {trainable_components}")

        # model declaration
        #config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
        #self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        if not encoder_trainable:
            _deactivate_relevant_gradients(model, trainable_components)

        # optimizer declaration
        #if optimizer == 'adam':
        #    self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        #elif optimizer == 'adamw':
        #    self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=True)
        #else:
        #    raise Exception(f"optimizer arg must be in ['adam', 'adamw'], got: {optimizer}")

        #self.learning_rate = learning_rate

        if verbose:
            print('\n\nTrainable Components:\n----------------------------------------\n')
            total_trainable_params = 0
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name, '  --->  ', param.shape)
                    total_trainable_params += param.shape[0] if len(param.shape) == 1 else param.shape[0] * param.shape[
                        1]
            print(
                f'\n----------------------------------------\nNumber of Trainable Parameters: {total_trainable_params}\n')


def _deactivate_relevant_gradients(model, trainable_components):
    """Turns off the model parameters requires_grad except the trainable_components.

    Args:
        trainable_components (List[str]): list of trainable components (the rest will be deactivated)

    """
    for param in model.parameters():
        param.requires_grad = False
    #if trainable_components:
    #    trainable_components = trainable_components + ['pooler.dense.bias']
    #trainable_components = trainable_components + ['classifier']
    trainable_components = trainable_components + ['ln_f']
    for name, param in model.named_parameters():
        for component in trainable_components:
            if component in name:
                param.requires_grad = True
                break

def convert_to_actual_components(components):
    return [BIAS_TERMS_DICT[component] for component in components]



def main():
    accelerator = Accelerator()
    model_name = "bigscience/bloom-7b1"
    MICRO_BATCH_SIZE = 1  # this could actually be 5 but i like powers of 2
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = 3  # we don't always need 3 tbh
    LEARNING_RATE = 3e-4  # the Karpathy constant
    CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    VAL_SET_SIZE = 2000
    DATA_PATH = "alpaca_data_cleaned.json"
    do_test = False
    OUTPUT_DIR = 'bloom-7b1-alpaca-bit'
    seed = 42
    set_seed(seed)
    accelerator.wait_for_everyone()

    model = BloomForCausalLM.from_pretrained( 
        model_name,
        #device_map='auto',
        #load_in_8bit=True,
    )
    tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom')

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    data = load_dataset("json", data_files=DATA_PATH)

    bias_terms = ['all']
    trainable_components = convert_to_actual_components(bias_terms)
    _perform_training_preparations(model, 'bitfit', trainable_components)


    def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
        user_prompt = (
            (
                f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Input:
    {data_point["input"]}

    ### Response:
    """
            )
            if data_point["input"]
            else (
                f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {data_point["instruction"]}

    ### Response:
    """
            )
        )
        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                    padding="max_length",
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]
        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }


    if VAL_SET_SIZE > 0:
        train_val = data["train"].train_test_split(
            test_size=VAL_SET_SIZE, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if VAL_SET_SIZE > 0 else None,
            save_steps=200,
            output_dir=OUTPUT_DIR, #output_dir=repository_id,
            save_total_limit=3,
            load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            #ddp_find_unused_parameters=False if ddp else None,
            torch_compile=True, # optimizations
            #optim="adamw_torch_fused", # improved optimizer
            optim="adamw_torch", # improved optimizer
            # push to hub parameters
            report_to='none',
            #push_to_hub=True,
            #hub_strategy="every_save",
            #hub_model_id=repository_id,
            #hub_token=HfFolder.get_token(),
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3


    trainer.train(resume_from_checkpoint = False)
    accelerator.wait_for_everyone()
    model.save_pretrained(OUTPUT_DIR)
    #torch.save(model.state_dict(), OUTPUT_DIR + '/model.pt')
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os
import random
import torch
from b import (
    no_tenfact,
    matlab_eigs
)
import numpy as np

import datasets
from datasets import load_metric

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from torch.utils.data import (
    DataLoader, 
    TensorDataset
)
import time
from data_processing import *
from datetime import datetime,timedelta

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return tensor_data, all_label_ids

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )    
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--local_rank',type=int, default=0, help='rank')
    parser.add_argument('--aug_train', action='store_true')
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def decoder(predicted_input, embedding_matrix):
    predicted_input = predicted_input.view(-1, predicted_input.shape[-1])
    for i in range(len(predicted_input)):
        if i > 15:
            break
        predict_token = predicted_input[i]
        best = -1
        best_dis = float("inf")
        for j in range(len(embedding_matrix)):
            candidate_token = embedding_matrix[j]
            dis = torch.norm(predict_token - candidate_token, p=2).item()
            if dis < best_dis:
                best = j
                best_dis = dis
        print(best)  
        
def main():
    args = parse_args()
    start = time.time()

    accelerator = Accelerator(fp16=False)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        logger.info(args)
        
    logger.info(accelerator.state)
    accelerator.wait_for_everyone()

    args.is_regression = args.task_name == "stsb"
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst2": Sst2Processor,
        "stsb": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst2": "classification",
        "stsb": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst2": {"num_train_epochs": 10, "max_seq_length": 64},
        "stsb": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    task_name = args.task_name
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    max_seq_length = default_params[task_name]["max_seq_length"]
    
    if not args.aug_train:
        train_examples = processor.get_train_examples(args.data_dir)
        eval_train_examples = train_examples
    else:
        train_examples = processor.get_aug_examples(args.data_dir)
        eval_train_examples = processor.get_train_examples(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, output_mode)
    train_dataset, train_labels = get_tensor_data(output_mode, train_features)

    eval_train_features = convert_examples_to_features(eval_train_examples, label_list, max_seq_length, tokenizer, output_mode)
    eval_train_dataset, eval_train_labels = get_tensor_data(output_mode, eval_train_features)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, output_mode)
    eval_dataset, eval_labels = get_tensor_data(output_mode, eval_features)

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=False, batch_size=args.per_device_train_batch_size
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    
    # model = BertForSequenceClassification(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True
    )

    model, train_dataloader = accelerator.prepare(
        model, train_dataloader
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")

    inputs = {}
    weights = {}
    def create_hook(name):
        def hook(model: torch.nn.Linear, input, output):
            if "classifier" in name or "pooler" in name:
                inputs[name] = input[0].detach().cpu().numpy()
                weights[name] = model.weight.data.detach().cpu().numpy()
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name))
    
    # m = 30000
    # d = 50/768
    # B = 1
    # Beta = 2
    # distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(50), covariance_matrix=torch.eye(50))
    # model.bert.pooler.dense.weight.data[:, :50] = distribution.sample((30000,))
    # model.bert.pooler.dense.weight.data[:, 50:] = 0
    # model.classifier.weight.data = torch.full((2, 30000), 1/30000).cuda()
    
    mse_loss = torch.nn.MSELoss(reduction='sum')
    for idx, batch in enumerate(train_dataloader):
        model.zero_grad()
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        input_embeddings = model.bert.embeddings.word_embeddings(input_ids)
        
        outputs1 = model(
            inputs_embeds=input_embeddings,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids,
        )
        
        loss = outputs1.loss
        dy_dx = torch.autograd.grad(loss, model.parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        #traverse model parameters and zero gradients
        model.zero_grad()
            
        dummpy_input = torch.randn((input_embeddings.shape)).cuda().requires_grad_(True)
        # dummy_label = torch.randn(label_ids.shape).cuda().requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummpy_input], max_iter=10, history_size=10, line_search_fn='strong_wolfe')

        print("Original loss: ", mse_loss(input_embeddings, dummpy_input).item())
        
        for i in range(300):
            def closure():
                optimizer.zero_grad()
                outputs2 = model(
                    inputs_embeds=dummpy_input,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    labels=label_ids,
                )
                loss_ce = outputs2.loss
                dummy_dy_dx = torch.autograd.grad(
                    loss_ce, 
                    model.parameters(), 
                    create_graph=True,
                    allow_unused=True
                )

                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
                    if gx is not None and gy is not None:
                        grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
            loss_mse = optimizer.step(closure)
            if i % 10 == 0:
                print(i, loss_mse.item())

        print("Reconstruct loss: ", mse_loss(input_embeddings, dummpy_input).item())
        decoder(dummpy_input, model.bert.embeddings.word_embeddings.weight)
        break
    end = time.time()
    print("Time: ", (end - start)/3600, "hours")

if __name__ == "__main__":
    main()

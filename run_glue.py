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

def evaluate_data(dataloader, dataset, model, args, mode="train"):
    model.eval()
    logger.info(f"***** Running {mode} evaluation *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")

    metric = None
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model( 
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids)
        predictions = outputs.logits.argmax(dim=-1) if not args.is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=label_ids,
        )
    eval_metric = metric.compute()
    model.train()
    return eval_metric

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
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
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
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
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument("--eval_step", default=200, type=int, help="eval step.")
    parser.add_argument("--print_step", default=10, type=int, help="print step.")
    parser.add_argument('--local_rank',type=int, default=0, help='rank')
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--do_eval',action='store_true')
    parser.add_argument('--early_stop',action='store_true')
    parser.add_argument('--early_stop_metric',default='accuracy', type=str, help="early stop metric")
    parser.add_argument('--save_last',action='store_true')
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

    args.lr_scheduler_type = SchedulerType[args.lr_scheduler_type.upper()]
    return args

def main():
    args = parse_args()
    start_time = time.time()

    es = None
    early_stop_trigger = False
    if args.early_stop:
        es = EarlyStopping(patience=100, mode='max')

    accelerator = Accelerator(fp16=True)

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(f"{args.output_dir}/last", exist_ok=True)

    if accelerator.is_main_process:
        logfilename = 'log_bs{}_lr{}_{}.txt'.format(args.per_device_train_batch_size, args.learning_rate, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        logfilename = os.path.join(args.output_dir, logfilename)
        handler = logging.FileHandler(logfilename)
        logger.addHandler(handler)
        logger.info('------------> log file =={}'.format(logfilename))
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

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

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
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
    eval_train_dataloader = DataLoader(eval_train_dataset, batch_size=args.per_device_eval_batch_size)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    
    # model = BertForSequenceClassification(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * 0.1),
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    tr_loss = 0

    inputs = {}
    weights = {}
    outputs = {}
    def create_hook(name):
        def hook(model: torch.nn.Linear, input, output):
            if "classifier" in name or "pooler" in name:
                inputs[name] = input[0].detach().cpu().numpy()
                # outputs[name] = output.detach().cpu().numpy()
                weights[name] = model.weight.data.detach().cpu().numpy()
        return hook

    handlers={}
    for name, layer in model.named_modules():
        if type(layer) == torch.nn.Linear:
            handlers[name] = layer.register_forward_hook(create_hook(name))
    args.max_train_steps = 1
    
    # import pdb
    # pdb.set_trace()
    
    distribution = torch.distributions.MultivariateNormal(loc=torch.zeros(50), covariance_matrix=torch.eye(50))
    model.bert.pooler.dense.weight.data[:, :50] = distribution.sample((30000,))
    model.bert.pooler.dense.weight.data[:, 50:] = 0

    model.classifier.weight.data = torch.full((2, 30000), 1/30000).cuda()

    if not args.do_eval:
        for epoch in range(args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.cuda() for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    labels=label_ids,
                )

                loss = outputs.loss * 1e-5
                # import pdb
                # pdb.set_trace()

                loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                accelerator.backward(loss)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # optimizer.step()
                    # lr_scheduler.step()
                    # optimizer.zero_grad()
                    completed_steps += 1

                if completed_steps % args.print_step == 0:
                    if accelerator.is_main_process:
                        logger.info("{:0>6d}/{:0>6d}, loss: {:.6f}, avg_loss: {:.6f}".format(
                            completed_steps,
                            args.max_train_steps, 
                            loss.item(),
                            tr_loss / completed_steps,
                            )
                        )

                # if completed_steps % args.eval_step == 0: 
                #     eval_metric = evaluate_data(
                #         eval_dataloader,
                #         eval_dataset,
                #         model,
                #         args,
                #         mode="dev"
                #     )
                #     logger.info(f"epoch {epoch}, step {completed_steps}/{args.max_train_steps}: {eval_metric}")

                    ##############################################################################
                    ##############################################################################
                    ##############################################################################
                    
                    # if args.early_stop:
                    #     assert args.early_stop_metric in eval_metric, "Early stop metric is not in evaluation result"
                    #     if es.step(eval_metric[args.early_stop_metric]):
                    #         early_stop_trigger = True
                    #         break
                    #     else:
                    #         if es.best == eval_metric[args.early_stop_metric]:
                    #             if args.output_dir is not None:
                    #                 accelerator.wait_for_everyone()
                    #                 unwrapped_model = accelerator.unwrap_model(model)
                    #                 unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                    #                 if accelerator.is_main_process:
                    #                     tokenizer.save_pretrained(args.output_dir) 
                    
                    ##############################################################################
                    ##############################################################################
                    ##############################################################################

                if completed_steps >= args.max_train_steps:
                    break
                
                if early_stop_trigger:
                    break
            
            if early_stop_trigger:
                break
            
            # if not args.early_stop:
            #     model.eval()
            #     if args.output_dir is not None:
            #         accelerator.wait_for_everyone()
            #         unwrapped_model = accelerator.unwrap_model(model)
            #         unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            #         if accelerator.is_main_process:
            #             tokenizer.save_pretrained(args.output_dir)   
                    
    # inputs, weights, outputs
    m = 30000
    d = 50
    B = 1
    Beta = 2
    
    # import pdb
    # pdb.set_trace()
    
    g = model.classifier.weight.grad.cpu().numpy()[1].reshape(m) # 1xm.
    W = weights["bert.pooler.dense"][:, :50] # m, d 
    
    import pickle
    with open('inputs.pickle', 'rb') as handle:
        others = pickle.load(handle)["bert.pooler.dense"]

    # with open('inputs.pickle', 'wb') as handle:
    #     pickle.dump(inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # import pdb
    # pdb.set_trace() 
    
    M = np.zeros((d, d))
    aa = np.sum(g)

    for i in range(d): #768
        for j in range(d): #768
            M[i, j] = np.sum(g * W[:, i] * W[:, j])
            if i == j:
                M[i, i] = M[i, i] - aa

    # pdb.set_trace()
    
    V, D = matlab_eigs(M, Beta)
    WV = W @ V

    T = np.zeros((Beta, Beta, Beta))
    for i in range(Beta):
        for j in range(i, Beta):
            for k in range(j, Beta):
                T[i, j, k] = np.sum(g * WV[:, i] * WV[:, j] * WV[:, k])
                T[i, k, j] = T[i, j, k]
                T[j, i, k] = T[i, j, k]
                T[j, k, i] = T[i, j, k]
                T[k, i, j] = T[i, j, k]
                T[k, j, i] = T[i, j, k]

    # pdb.set_trace()
    
    for i in range(Beta):
        for j in range(Beta):
            aa = np.sum(g * WV[:, i])
            T[i, j, j] = T[i, j, j] - aa
            T[j, i, j] = T[j, i, j] - aa
            T[j, j, i] = T[j, j, i] - aa

    T = T / m
    rec_X, _, misc = no_tenfact(T, 100, B)
    new_recX = V @ rec_X
    
    # pdb.set_trace()
    
    new_recXX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    from scipy.spatial import distance
    input = inputs["bert.pooler.dense"][:, :50]
    print(input.shape)
    print(new_recXX.shape)
    print(f"cosin similarity: {1-distance.cosine(new_recXX.reshape(-1), input.reshape(-1))}",
          f"normalized error: {np.sum((new_recXX.reshape(-1) - input.reshape(-1))**2)}")
    print("#" * 50)
    print("#" * 50)
    for idx in range(len(others)):
        print(f"cosin similarity: {1-distance.cosine(new_recXX.reshape(-1), others[idx][:50].reshape(-1))}", 
              f"normalized error: {np.sum((new_recXX.reshape(-1) - others[idx][:50])**2)}")
        
    for i in range(B):
        new_recX[:, i] = -new_recX[:, i]

    new_recX = (new_recX / np.linalg.norm(new_recX, ord=2, axis=0)).transpose()
    from scipy.spatial import distance
    input = inputs["bert.pooler.dense"][:, :50]
    print(input.shape)
    print(new_recX.shape)
    print(f"cosin similarity: {1-distance.cosine(new_recX.reshape(-1), input.reshape(-1))}",
          f"normalized error: {np.sum((new_recX.reshape(-1) - input.reshape(-1))**2)}")
    print("#" * 50)
    print("#" * 50)
    for idx in range(len(others)):
        print(f"cosin similarity: {1-distance.cosine(new_recX.reshape(-1), others[idx][:50].reshape(-1))}", 
              f"normalized error: {np.sum((new_recX.reshape(-1) - others[idx][:50])**2)}")
    
    import pdb
    pdb.set_trace()
    
    # eval_metric = evaluate_data(
    #     eval_train_dataloader,
    #     eval_train_dataset,
    #     model,
    #     args,
    #     mode="train"
    # )
    # logger.info(f"Train Dataset Result: {eval_metric}")      

    # eval_metric = evaluate_data(
    #     eval_dataloader,
    #     eval_dataset,
    #     model,
    #     args,
    #     mode="dev"
    # )
    # logger.info(f"Dev Dataset Result: {eval_metric}")  

    # if args.early_stop and es:
    #     logger.info(f"DEV Best Result: {args.early_stop_metric}, {es.best}")

    # if args.output_dir is not None and args.save_last:
    #     accelerator.wait_for_everyone()
    #     unwrapped_model = accelerator.unwrap_model(model)
    #     unwrapped_model.save_pretrained(f"{args.output_dir}/last", save_function=accelerator.save)
    #     if accelerator.is_main_process:
    #         tokenizer.save_pretrained(f"{args.output_dir}/last")
            
    # if accelerator.is_main_process:
    #     total_time = time.time() - start_time
    #     total_time_str = str(timedelta(seconds=int(total_time)))
    #     logger.info('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    main()

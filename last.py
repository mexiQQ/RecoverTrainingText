from transformers import AutoTokenizer, OPTForSequenceClassification
import torch
from datasets import load_dataset
from tqdm import tqdm
# from b import no_tenfact
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = OPTForSequenceClassification.from_pretrained("facebook/opt-125m").cuda()

# tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-STS-B")
# model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-STS-B").cuda()

import pdb
pdb.set_trace()

def evaluation(model, tokenizer, split='test', num_sentences=1024, batch_size=1):
    device = model.device
    max_length = model.config.max_position_embeddings
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    stride = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    counter = 0
    
    batch_input = []
    batch_target = []
    for begin_loc in tqdm(range(0, seq_len, stride)):
        counter += 1
        
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        batch_input.append(input_ids)
        batch_target.append(target_ids)
        
        if len(batch_input) < batch_size:
            continue
        else:
            inputs = torch.stack(batch_input, dim=0).squeeze(1)
            targets = torch.stack(batch_target, dim=0).squeeze(1)
            
        with torch.no_grad():
            outputs = model(inputs, labels=targets)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        
        batch_input = []
        batch_target = []

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
        if counter >= num_sentences:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(f'perplexity: {ppl}')


def get_inputs_and_labels(model, tokenizer, split='test', batch_size=1):
    device = model.device
    max_length = model.config.max_position_embeddings
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    stride = 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    counter = 0
    
    batch_input = []
    batch_target = []
    for begin_loc in tqdm(range(0, seq_len, stride)):
        counter += 1
        
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        batch_input.append(input_ids)
        batch_target.append(target_ids)
        
        if len(batch_input) < batch_size:
            continue
        else:
            inputs = torch.stack(batch_input, dim=0).squeeze(1)
            targets = torch.stack(batch_target, dim=0).squeeze(1)
            return inputs, targets

    
hook_storage = {}
def create_hook(name):
    def hook(model: torch.nn.Linear, input, output):
        if "lm_head" in name:
            hook_storage[name] = input
    return hook

handlers={}
for name, layer in model.named_modules():
    if type(layer) == torch.nn.Linear:
        handlers[name] = layer.register_forward_hook(create_hook(name))

# evaluation(model=model, tokenizer=tokenizer, split='train', num_sentences=10, batch_size=10)

inputs, targets = get_inputs_and_labels(model, tokenizer=tokenizer, split='train', batch_size=1)

outputs = model(inputs, labels=targets)

loss = outputs.loss
logits = outputs.logits

loss.backward()

lm_head_gradient = model.lm_head.weight.grad # [50272, 768]
import pdb
pdb.set_trace()

# 1.
# regression => STS-B glue => mse loss
# Bert downstream task
# 5 分制度

# 2.
# 768 x 50000 (vocabluraly)
# activation function (gelu) 

# 3. 
# a(50000) (768 x 50000 + nolinear)
# B x sequence length x 768

# => 768 x 50000 linear

# => gelu

# => B x sequence length x 1 logits

# => Mse loss

# =====
#
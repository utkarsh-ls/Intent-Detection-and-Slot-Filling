import torch 
from torch import nn
from model import TorchGRUIntent, TorchLSTMIntent, TransformerIntent, SupConLoss
from utils import Trainer, EarlyStopping
import argparse
import os
from sklearn.utils import class_weight

parser = argparse.ArgumentParser(description='Get configurations to train')
parser.add_argument('--accent', default='hindi_female', type=str)
parser.add_argument('--lang', default='en', type=str)
# parser.add_argument('--orig', default=False, type=bool)
parser.add_argument('--test_accent', default='hindi_female', type=str)
parser.add_argument('--cpu_cores', default=8, type=int)
parser.add_argument('--data', default="~", type=str)
parser.add_argument('--model_type', default="TGRU", type=str)
parser.add_argument('--model', default="", type=str)
parser.add_argument('--mode', default="train", type=str)
parser.add_argument('--slot', default=True, type=bool)
CONFIG = parser.parse_args()

if CONFIG.lang == 'hi':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
else:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device: ", device)

if CONFIG.slot==True:
    from dataset import BTP3 as BTP
    model_mode = 'slot'
else:
    from dataset import BTP2 as BTP
    model_mode = 'intent'

checkpoint_dir = '/scratch/ut_ckp'
# checkpoint_dir = os.readlink(checkpoint_dir)
checkpoint_dir += f'/{CONFIG.accent}/'
os.makedirs(checkpoint_dir, exist_ok=True)
m_dir = 'models/'
os.makedirs(m_dir, exist_ok=True)

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
epochs = 50
cpu_cores = CONFIG.cpu_cores
print(f"Using {cpu_cores} CPU cores")

# Getting dataset and data loaders
data_dir = CONFIG.data
if data_dir[-1] != "/":
    data_dir += "/"
model_name = f"TGRU_{model_mode}_{CONFIG.lang}_{CONFIG.accent}_contrast_out_{batch_size}_0.1"

train_dataset = BTP("train", accent=CONFIG.accent, lang=CONFIG.lang, mode=model_mode)
val_dataset = BTP("validation", accent=CONFIG.accent, lang=CONFIG.lang, mode=model_mode)
test_dataset = BTP("test", accent=CONFIG.test_accent, lang=CONFIG.lang, mode=model_mode)
if model_mode=='slot':
    pad_index = train_dataset.slot_map["O"]
    pad_weight = (train_dataset.slot_count-train_dataset.empty_slot_count)/(train_dataset.empty_slot_count)
    print(f'{train_dataset.slot_count}, {train_dataset.empty_slot_count}')
else:
    pad_index = -100
    pad_weight = 1

def batch_sequences(seq_list):
    token_ids = []
    targets = []
    if model_mode=='slot':
        length_sum = 0
        for ids, slots, length in seq_list:
            token_ids.append(ids)
            targets.append(slots)
            length_sum += length
        # token_ids = nn.utils.rnn.pad_sequence(token_ids, padding_value=BTP.pad_token_id, batch_first=True)
        token_ids = torch.stack(token_ids)
        targets = torch.stack(targets)
        attention_mask = (token_ids != BTP.pad_token_id)
        # print('token_ids: ', token_ids.shape)
        # print('targets: ', targets.shape)
        # exit(0)
        return token_ids, attention_mask, targets, length_sum
    else:
        scenarios = [] 
        for ids, labels, scenario in seq_list:
            token_ids.append(ids)
            targets.append(labels)
            scenarios.append(scenario)

        token_ids = nn.utils.rnn.pad_sequence(token_ids, padding_value=BTP.pad_token_id, batch_first=True)
        targets = torch.stack(targets)
        scenarios = torch.stack(scenarios)
        attention_mask = (token_ids != BTP.pad_token_id)
        return token_ids, attention_mask, targets, scenarios


train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

val_loader = torch.utils.data.DataLoader(val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False, 
                                            num_workers=cpu_cores,
                                            collate_fn=batch_sequences)

# Getting the new model
if CONFIG.model_type == "TGRU":
    if model_mode == 'slot':
        model = TorchGRUIntent(hidden_size=300, vocab_size=len(train_dataset.slot_map), scenario_size=train_dataset.scenario_count)
    else:
        model = TorchGRUIntent(hidden_size=300, vocab_size=train_dataset.intent_count, scenario_size=train_dataset.scenario_count)
elif CONFIG.model_type == "TLSTM":
    model = TorchLSTMIntent(hidden_size=300, vocab_size=train_dataset.intent_count)
elif CONFIG.model_type == "Transformer":
    model = TransformerIntent(vocab_size=train_dataset.intent_count)
else:
    print("Unidentified model type")
    exit(1)

# Model Path
CONFIG.model = '/scratch/ut_ckp/final/' + model_name + 'final.pt'

if CONFIG.mode == 'train':
    print("Training new model ", type(model))
else:
    print("Using model from", CONFIG.model)
    model.load_state_dict(torch.load(CONFIG.model))
    model = model.to(device)


# Optimizer and Criterion

class_weights = [1]*len(train_dataset.slot_map)
class_weights[pad_index] = pad_weight
class_weights = torch.tensor(class_weights,dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                        mode='min', 
                                                        factor=0.5, 
                                                        patience=4, 
                                                        verbose=True)  


# Early Stopping
early_stopping = EarlyStopping(patience=5)

CONFIG.accent = f'{CONFIG.accent}_{CONFIG.lang}' if CONFIG.accent=='orig' else CONFIG.accent

# Train the model
trainer = Trainer(model_name=model_name,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    checkpoint_dir=checkpoint_dir,
                    mode=model_mode,
                    early_stopping=early_stopping,
                    log_periodicity=25,
                    checkpoint_strategy="periodic",
                    checkpoint_periodicity=1,
                    pad_index=pad_index,
                    vocab_size=len(train_dataset.slot_map),
                    test_loader=test_loader)

if CONFIG.mode == "train":
    trainer.train()

# Test
trainer.evaluate(name="Val", loader=val_loader)
trainer.evaluate(name="Test", loader=test_loader)

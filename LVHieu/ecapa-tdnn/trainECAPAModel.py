
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from dataLoader import trainDataset
from ECAPAModel import ECAPA_TDNN
from torch.utils.data import DataLoader
from lossFunction import AAMsoftmax
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, required=True)
parser.add_argument('--eval_list', type=str, default='')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--initial_model', type=str, default='')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--noise_level', type=float, default=0.005, help="Mức độ noise trắng thêm vào")
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ECAPA_TDNN(C=1024).to(device)
aamsoftmax_layer = AAMsoftmax(n_class=5000).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(list(model.parameters()) + list(aamsoftmax_layer.parameters()),
                      lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

if args.initial_model != "":
    print("Loading initial model:", args.initial_model)
    checkpoint = torch.load(args.initial_model, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("speaker_encoder."):
            new_key = k.replace("speaker_encoder.", "")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

train_dataset = trainDataset(args.train_list, args.data_path, noise_level=args.noise_level)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

print("Start Training")
for epoch in range(1, args.epochs + 1):
    model.train()
    aamsoftmax_layer.train()
    running_loss = 0.0
    for idx, (audios, labels) in enumerate(train_loader):
        audios = audios.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(audios, aug=True)
        logits = aamsoftmax_layer(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 100 == 0:
            avg_loss = running_loss / (idx + 1)
            print(f"Epoch {epoch} [{idx}/{len(train_loader)}] Avg Loss: {avg_loss:.4f}")

    scheduler.step()
    torch.save({'model': model.state_dict(),
                'aamsoftmax': aamsoftmax_layer.state_dict()}, os.path.join(args.save_path, f'model_{epoch}.pt'))

torch.save({'model': model.state_dict(),
            'aamsoftmax': aamsoftmax_layer.state_dict()}, os.path.join(args.save_path, 'final.model'))



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.datasets.librispeech as lr
import pathlib as pl
import torch.utils.data as utils
import torchaudio
import sklearn.model_selection as sk
import os
from tqdm.notebook import tqdm


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(train_loader)
        # print(type(train_loader))
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        # pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())


def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, epoch, labels):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        # pbar.update(pbar_update)

    print(
        #({100. * correct / len(test_loader.dataset):.0f}%) because not all waveforms are applicable
        f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} \n")

    def predict(tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = transform(tensor)
        tensor = model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        tensor = labels.index((tensor.squeeze()))

        return tensor


def label_to_index(int_arr):
    pass


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id in batch:
        wave_len = len(waveform.detach().numpy().flatten())

        if (wave_len >= sample_rate * 3) and (wave_len <= sample_rate * 5):
            tensors += [waveform]
            # print(speaker_id)
            # print(label_index[speaker_id])
            targets += [torch.tensor(labels.index(speaker_id))]
            # print(targets)

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


if __name__ == "__main__":

    full_set = lr.LIBRISPEECH(root="C:/Users/op370/PycharmProjects/TaskSAMSUNG", url="dev-clean", download=True)

    train_size = int(0.90 * len(full_set))
    test_size = len(full_set) - train_size
    train_set, test_set = torch.utils.data.random_split(full_set, [train_size, test_size])

    print(len(train_set))
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = train_set[5]
    print("Shape of waveform: {}".format(waveform.size()))
    print("speaker: {}".format(speaker_id))
    print("speaker: {}".format(chapter_id))
    print("utterance: {}".format(utterance_id))
    print("Sample rate of waveform: {}".format(sample_rate))

    label_index = {}

    labels = sorted(list(set(datapoint[3] for datapoint in train_set)))
    print(labels)
    print("labels len = " + str(len(labels)))
    for label in labels:
        label_index[label] = labels.index(label)

    # data = sk.train_test_split(train_set, train_size=0.8, test_size=0.2, shuffle=True) #stratify=labels)
    # train = data[0]
    # test_set = data[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 126

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    transformed = transform(waveform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = M5(n_input=transformed.shape[0], n_output=len(labels))
    model.to(device)
    print(model)


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    n = count_parameters(model)
    print("Number of parameters: %s" % n)

    optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                          gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10

    log_interval = 20
    n_epoch = 12

    # pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transform.to(device)
    # with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        test(model, epoch, labels)
        scheduler.step()
    torch.save(model.state_dict(), "C:/Users/op370/PycharmProjects/TaskSAMSUNG/model2.pth")

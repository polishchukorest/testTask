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
import random


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


def predict(model, tensor, device, labels):
    transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=8000)
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = model(tensor.unsqueeze(0))
    if get_likely_index(tensor) == -1:
        return torch.tensor(-1)
    tensor = get_likely_index(tensor)
    # print((tensor.squeeze()))
    print(int((tensor.squeeze())))
    tensor = torch.tensor(labels[tensor.squeeze()])
    return tensor


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    index_max = tensor.argmax(dim=-1)
    if tensor.detach().numpy().flatten()[index_max] < 0:
        return -1
    return index_max


if __name__ == "__main__":
    train_set = lr.LIBRISPEECH(root="C:/Users/op370/PycharmProjects/TaskSAMSUNG", url="dev-clean", download=False)
    other_set = lr.LIBRISPEECH(root="C:/Users/op370/PycharmProjects/TaskSAMSUNG", url="test-other", download=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = sorted(list(set(datapoint[3] for datapoint in train_set)))
    print(f"labels = {labels}")
    model = M5(1, 40)
    state_dict = torch.load("C:/Users/op370/PycharmProjects/TaskSAMSUNG/model2.pth")
    model.load_state_dict(state_dict)
    score = 0

    # print(f"Expected: {speaker_id}. Predicted: {predict(model, waveform, device, labels)}.")
    count_cases_train = 0
    count_cases_other = 0
    count_frr = 0
    count_far = 0

    print("---" * 10)
    for i in range(50):

        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = train_set[(random.randint(1, 500))]
        waveform_, sample_rate_, utterance_, speaker_id_, chapter_id_, utterance_id_ = other_set[(random.randint(1, 50))]
        wave_len = len(waveform.detach().numpy().flatten())
        wave_len_ = len(waveform_.detach().numpy().flatten())
        if (wave_len >= sample_rate * 3) and (wave_len <= sample_rate * 5):
            count_cases_train += 1
            if speaker_id == predict(model, waveform, device, labels):
                score += 1
            else:
                count_frr += 1
        if (wave_len_ >= sample_rate_ * 3) and (wave_len_ <= sample_rate_ * 5):
            count_cases_other += 1
            if predict(model, waveform_, device, labels) != -1:
                count_far += 1

    print(f"train_checks: {count_cases_train}, score = {score}, frr = {count_frr}/{count_cases_train}")
    print(f"other_checks: {count_cases_other}, far = {count_frr}/{count_cases_other} ")

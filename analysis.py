import torch.nn as nn

class CutInDetectionModel(nn.Module):
    def _init_(self, input_size, hidden_size, num_layers, num_classes):
        super(CutInDetectionModel, self)._init_()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Parameters
input_size = 128  # Adjust based on the feature vector size
hidden_size = 64
num_layers = 2
num_classes = 2  # Binary classification: Cut-in or not

# Initialize the model
model = CutInDetectionModel(input_size, hidden_size, num_layers, num_classes)
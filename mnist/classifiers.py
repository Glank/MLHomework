import torch
import dataset

class SimpleLinearModel(torch.nn.Module):
  def __init__(self):
    super(SimpleLinearModel, self).__init__()
    self.sequence = torch.nn.Sequential(
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(28*28, 10)
    )

  def forward(self, x):
      return self.sequence(x)

class FullyConnectedNNModel(torch.nn.Module):
  def __init__(self):
    super(FullyConnectedNNModel, self).__init__()
    self.sequence = torch.nn.Sequential(
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(28*28, 100),
      torch.nn.ReLU(),
      torch.nn.BatchNorm1d(100),
      torch.nn.Linear(100, 30),
      torch.nn.ReLU(),
      torch.nn.Linear(30, 10),
    )

  def forward(self, x):
      return self.sequence(x)

class ConvolutionalNNModel(torch.nn.Module):
  def __init__(self):
    super(ConvolutionalNNModel, self).__init__()
    img_width = 28
    conv1_out_width = (img_width-4)/1+1
    conv2_out_width = (conv1_out_width-3)/1+1
    conv3_out_width = (conv2_out_width-3)/1+1
    linear_width = int(conv3_out_width*conv3_out_width*2)
    print(linear_width)
    self.sequence = torch.nn.Sequential(
      torch.nn.Conv2d(1, 5, 4, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(5),
      torch.nn.Conv2d(5, 10, 3, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(10),
      torch.nn.Conv2d(10, 2, 3, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(2),
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(linear_width, 30),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(30, 10),
    )

  def forward(self, x):
      return self.sequence(x)


def main():
  ds = dataset.MNISTDataset()
  batch_size = 64
  train_data = torch.utils.data.DataLoader(ds, batch_size=batch_size)

  #device = "cuda" if torch.cuda.is_available() else "cpu"
  device = "cpu"
  print(f"Using {device} device")

  #model = SimpleLinearModel()
  #model = FullyConnectedNNModel()
  model = ConvolutionalNNModel()
  
  #loss_fn = torch.nn.MSELoss(reduction='sum')
  loss_fn = torch.nn.CrossEntropyLoss()
  #optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)

  for t in range(1):
    for batch, (X, y) in enumerate(train_data):
      y_pred = model(X.unsqueeze(1))
      loss = loss_fn(y_pred, y)
      print(t, batch, loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  model.eval()
  # test on a few samples
  to_test = set(range(0, 5))
  to_test.update(range(5000, 5005))
  for i, (X, y) in enumerate(ds):
    if i not in to_test:
      continue

    y_pred = model(X.unsqueeze(0).unsqueeze(0))
    label_pred = torch.argmax(y_pred)
    print(f'Predicted label: {label_pred}')
    label = torch.argmax(y)
    print(f'Real label: {label}')
    dataset.print_img(X)

if __name__=='__main__':
  main()

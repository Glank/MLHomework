import torch
import dataset
from itertools import chain

class Discriminator(torch.nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    img_width = 28
    conv1_out_width = (img_width-3)/1+1
    conv2_out_width = (conv1_out_width-3)/1+1
    conv3_out_width = (conv2_out_width-3)/1+1
    linear_width = conv3_out_width*conv3_out_width*10
    print(linear_width)
    linear_width = int(linear_width)
    self.sequence = torch.nn.Sequential(
      torch.nn.Conv2d(1, 5, 3, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(5),
      torch.nn.Conv2d(5, 10, 3, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(10),
      torch.nn.Conv2d(10, 10, 3, stride=1),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(10),
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(linear_width, 500),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm1d(500),
      torch.nn.Linear(500, 200),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm1d(200),
      torch.nn.Linear(200, 100),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm1d(100),
      torch.nn.Linear(100, 2),
      torch.nn.LeakyReLU(),
    )

  def forward(self, x):
    return self.sequence(x)

class Generator(torch.nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    rand_input_width = 10
    convtran1_out_width = (rand_input_width-1)*2-2*0+1*(4-1)+0+1
    print(convtran1_out_width)
    convtran2_out_width = (convtran1_out_width-1)*1-2*0+1*(4-1)+0+1
    print(convtran2_out_width)
    convtran3_out_width = (convtran2_out_width-1)*1-2*0+1*(4-1)+0+1
    print(convtran3_out_width)
    conv1_out_width = (convtran3_out_width-1)/1+1
    print(conv1_out_width)
    self.sequence = torch.nn.Sequential(
      torch.nn.ConvTranspose2d(10, 5, 4, stride=2),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(5),
      torch.nn.ConvTranspose2d(5, 3, 4, stride=1),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.BatchNorm2d(3),
      torch.nn.ConvTranspose2d(3, 3, 4, stride=1),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(3, 1, 1),
      torch.nn.LeakyReLU(),
    )

  def forward(self, x):
    return self.sequence(x)*256

class StupidGenerator(torch.nn.Module):
  def __init__(self):
    super(StupidGenerator, self).__init__()
    downsample_size = 7*7 
    self.choice_sequence = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(1000, 75),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(75, 50),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(50,50),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(50, downsample_size*3),
      torch.nn.LeakyReLU(),
    )
    
    self.upsample_sequence = torch.nn.Sequential(
      torch.nn.Conv2d(3, 5, 3, stride=1, padding=1),
      torch.nn.LeakyReLU(),
      torch.nn.Upsample(scale_factor=2, mode='bilinear'),
      torch.nn.Conv2d(5, 10, 3, stride=1, padding=1),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Upsample(scale_factor=2, mode='bilinear'),
      torch.nn.Conv2d(10, 8, 3, stride=1, padding=1),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(8, 5, 3, stride=1, padding=1),
      torch.nn.Dropout(),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(5, 1, 3, stride=1, padding=1),
      torch.nn.LeakyReLU(),
    )

  def forward(self, x):
    choice = self.choice_sequence(x).reshape(x.shape[0], 3, 7, 7)
    return self.upsample_sequence(choice)*255

def print_img(img):
  rng = (torch.min(img), torch.max(img))
  img = 255*(img-rng[0])/(rng[1]-rng[0])
  chars = list(' .;-+=O#@')
  for r in range(img.shape[0]):
    for c in range(img.shape[1]):
      print(chars[int(img[r,c]/256.*len(chars))], end='')
    print('')

def main():
  ds = dataset.MNISTDataset(only_label=8)
  batch_size = 64
  train_data = torch.utils.data.DataLoader(ds, batch_size=batch_size)

  device = "cpu"
  print(f"Using {device} device")

  generator = StupidGenerator()
  discriminator = Discriminator()
  
  disc_loss_fn = torch.nn.CrossEntropyLoss()
  gen_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=0.5e-2, weight_decay=1e-2)
  disc_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=1e-2, weight_decay=1e-2)

  for t in range(1000):
    for batch, (X, y) in enumerate(train_data):
      gen_exp_lbls = torch.zeros(X.shape[0], dtype=torch.long)
      disc_exp_lbls = torch.zeros(X.shape[0]*2, dtype=torch.long)
      disc_exp_lbls[X.shape[0]:X.shape[0]*2] = 1

      print('Epoc {}, Batch {}:'.format(t, batch))

      gen_seed = torch.rand((X.shape[0], 10, 10, 10))
      gen_out = generator(gen_seed)
      disc_out_gen = discriminator(gen_out)
      gen_loss = disc_loss_fn(disc_out_gen, gen_exp_lbls)
      print('  gen_loss: {}'.format(gen_loss))

      noisy_X = (X+torch.rand(X.shape)*10)
      disc_out_X = discriminator(noisy_X.unsqueeze(1))
      disc_out = torch.cat((disc_out_X, disc_out_gen))
      disc_loss = disc_loss_fn(disc_out, disc_exp_lbls)
      print('  disc_loss: {}'.format(disc_loss))
      t1_errors = torch.sum(torch.argmax(disc_out_X, dim=1))
      print('  t1 disc errors: {}'.format(t1_errors))
      t2_errors = X.shape[0]-torch.sum(torch.argmax(disc_out_gen, dim=1))
      print('  t2 disc errors: {}'.format(t2_errors))

      if disc_loss > 0.5:
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
      else:
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        print('Real:')
        print_img(noisy_X[0,:,:])
        print('Fake:')
        print_img(gen_out[0,0,:,:])
  exit()

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

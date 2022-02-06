import torch
from itertools import chain

class Discriminator(torch.nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.sequence = torch.nn.Sequential(
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(4, 20),
      torch.nn.LeakyReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(20, 5),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(5, 2),
      torch.nn.LeakyReLU(),
    )

  def forward(self, x):
    return self.sequence(x)

class Generator(torch.nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.sequence = torch.nn.Sequential(
      torch.nn.Flatten(1, -1),
      torch.nn.Linear(2*2*3, 20),
      torch.nn.LeakyReLU(),
      torch.nn.Dropout(),
      torch.nn.Linear(20, 10),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(10, 10),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(10, 2*2),
      torch.nn.ReLU(),
    )

  def forward(self, x):
    return (self.sequence(x)*256).reshape((x.shape[0], 1, 2, 2))

class BabyDataset(torch.utils.data.IterableDataset):
  def __init__(self, start, end):
    self.start = start
    self.end = end
  def __iter__(self):
    for _ in range(self.start, self.end):
      x = torch.rand((2,2));
      x = (x*10)+torch.tensor([
        [245, 118],
        [118, 0],
      ])
      yield x

def print_img(img):
  img = torch.clamp(img, 0, 255)
  chars = list(' .;-+=O#@')
  for r in range(img.shape[0]):
    for c in range(img.shape[1]):
      print(chars[int(img[r,c]/256.*len(chars))], end='')
    print('')

def main():
  ds = BabyDataset(0, 6000)
  batch_size = 64
  train_data = torch.utils.data.DataLoader(ds, batch_size=batch_size)

  device = "cpu"
  print(f"Using {device} device")

  generator = Generator()
  discriminator = Discriminator()
  
  disc_loss_fn = torch.nn.CrossEntropyLoss()
  gen_optimizer = torch.optim.Adam(
    generator.parameters(),
    lr=1e-3, weight_decay=1e-2)
  disc_optimizer = torch.optim.Adam(
    discriminator.parameters(),
    lr=0.5e-2, weight_decay=1e-2)

  for t in range(10):
    for batch, X in enumerate(train_data):
      gen_exp_lbls = torch.zeros(X.shape[0], dtype=torch.long)
      disc_exp_lbls = torch.zeros(X.shape[0]*2, dtype=torch.long)
      disc_exp_lbls[X.shape[0]:X.shape[0]*2] = 1

      print('Epoc {}, Batch {}:'.format(t, batch))

      gen_seed = torch.rand((X.shape[0], 3, 2, 2))
      gen_out = generator(gen_seed)
      disc_out_gen = discriminator(gen_out)
      gen_loss = disc_loss_fn(disc_out_gen, gen_exp_lbls)
      print('  gen_loss: {}'.format(gen_loss))
      gen_min = torch.min(gen_out[0,0,:,:])
      gen_max = torch.max(gen_out[0,0,:,:])
      print('  gen 0,0 betwee [{}, {}]'.format(gen_min, gen_max))

      disc_out_X = discriminator(X.unsqueeze(1))
      disc_out = torch.cat((disc_out_X, disc_out_gen))
      disc_loss = disc_loss_fn(disc_out, disc_exp_lbls)
      print('  disc_loss: {}'.format(disc_loss))
      t1_errors = torch.sum(torch.argmax(disc_out_X, dim=1))
      print('  t1 disc errors: {}'.format(t1_errors))
      t2_errors = X.shape[0]-torch.sum(torch.argmax(disc_out_gen, dim=1))
      print('  t2 disc errors: {}'.format(t2_errors))

      print('real:')
      print_img(X[0,:,:])
      print('fake:')
      print_img(gen_out[0,0,:,:])

      if disc_loss > 0.5:
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
      else:
        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
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

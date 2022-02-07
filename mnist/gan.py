import torch
import dataset
from itertools import chain
import tempfile
import io

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

class GanTrainingScheduler:
  def __init__(self):
    self.dataset = dataset.MNISTDataset(only_label=8)
    self.dataset_iter = None
    self.batch_size = 64
    self.counter_example_cache = tempfile.TemporaryFile()
    self.counter_examples = []
    # Each element of counter_examples
    # {
    #   file_idx - the seekable index in counter_example_cache
    #   loss - the loss as of the last update
    #   last_update - the number of training iterations performed by the discriminator before saving this counter example
    #   retention_rate - affects how quickly loss increases at times beyond the last_update
    # }
    self.disc_baseline_retention_rate = .95
    self.disc_current_batch_index = 0
    self.disc_loss_cuttoff = 0.5 # when the predicted loss exceeds this, an example will be returned 
    self.max_counter_examples = 32
    self.counter_example_index = 0
  def next_training_samples(self):
    # circularly select counter examples
    counter_examples = []
    countre_examples_indicies = []
    i = self.counter_example_index
    visited = 0
    while visited < len(counter_examples):
      visited+=1
      if self._predict_loss(i) > self.disc_loss_cuttoff:
        counter_examples.append(self._load_counter_example(i))
        counter_examples_indicies.append(i)
      i = (i+1)%len(self.counter_examples)
      if len(counter_examples) >= self.max_counter_examples:
        break
    self.counter_example_index = i

    real_samples = []
    act_batch_size = self.batch_size + len(counter_examples)
    if self.dataset_iter is None:
      self.dataset_iter = iter(self.dataset)
    while len(real_samples) < act_batch_size:
      try:
        X, y = next(self.dataset_iter)
        real_samples.append(X)
      except StopIteration:
        self.dataset_iter = iter(self.dataset)

    return (
      torch.cat(s.unsqueeze(0) for s in real_samples),
      torch.cat(c.unsqueeze(0) for c in counter_examples),
      counter_examples_indicies,
    )
  def update_disc_results(gan_samples, gan_samples_loss, counter_examples_indicies, counter_examples_loss):
    self.disc_current_batch_index += 1
    i = torch.argmax(gan_samples_loss)
    self._save_counter_example(gan_samples[i], gan_samples_loss[i])
    for i in counter_examples_indicies:
      self._update_counter_example(i, counter_examples_loss[i])
  def _predict_loss(self, index):
    entry = self.counter_examples[index]
    p_correct = math.exp(-entry['loss'])
    age = self.disc_current_batch_index = entry['last_update']
    p_correct = math.pow(entry['retention_rate'], age)*p_correct
    return -math.log(p_correct)
  def _update_counter_example(self, index, new_real_loss):
    entry = self.counter_examples[index]
    old_p_correct = math.exp(-entry['loss'])
    entry['loss'] = new_real_loss
    new_p_correct = math.exp(-new_real_loss)
    age = self.disc_current_batch_index = entry['last_update']
    if age > 10:
      # new_p_correct = old_p_correct * (real_retention_rate ^ age)
      # (new_p_correct / old_p_correct) = real_retention_rate ^ age
      # (new_p_correct / old_p_correct) ^ (1/age) = real_retention_rate
      entry['retention_rate'] = max(0.001, min(0.999, math.pow(new_p_correct / old_p_correct, 1/age)))
      running_avg_pool_size = min(len(self.counter_examples), 100)
      self.disc_baseline_retention_rate = (self.disc_baseline_retention_rate*(running_avg_pool_size-1) + entry['retention_rate'])/running_avg_pool_size
  def _save_counter_example(self, example, loss):
    entry = {}
    entry['loss'] = loss
    f = self.counter_example_cache
    f.seek(0, io.SEEK_END)
    entry['file_idx'] = f.tell()
    torch.save(example, f)
    entry['last_update'] = self.disc_current_batch_index
    entry['retention_rate'] = self.disc_baseline_retention_rate
    self.counter_examples.append(entry)
  def _load_counter_example(self, index):
    entry = self.counter_examples[index]
    f = self.counter_example_cache
    f.seek(entry['file_idx'])
    return torch.load(f)

def set_lr(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def main():
  ds = dataset.MNISTDataset(only_label=8)
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
    lr=1e-4, weight_decay=1e-2)

  for t in range(1000):
    for batch, (X, y) in enumerate(train_data):
      gen_exp_lbls = torch.zeros(X.shape[0], dtype=torch.long)
      disc_exp_gen_lbls = torch.ones(X.shape[0], dtype=torch.long)
      disc_exp_real_lbls = torch.zeros(X.shape[0], dtype=torch.long)

      print('Epoc {}, Batch {}:'.format(t, batch))

      gen_seed = torch.rand((X.shape[0], 10, 10, 10))
      gen_out = generator(gen_seed)
      disc_out_gen = discriminator(gen_out)
      gen_loss = disc_loss_fn(disc_out_gen, gen_exp_lbls)
      print('  gen_loss: {}'.format(gen_loss))

      noisy_X = (X+torch.rand(X.shape)*10)
      disc_out_X = discriminator(noisy_X.unsqueeze(1))
      disc_loss = \
        0.5*disc_loss_fn(disc_out_X, disc_exp_real_lbls) + \
        0.5*disc_loss_fn(disc_out_gen, disc_exp_gen_lbls)
      print('  disc_loss: {}'.format(disc_loss))
      t1_errors = torch.sum(torch.argmax(disc_out_X, dim=1))
      print('  t1 disc errors: {}'.format(t1_errors))
      t2_errors = X.shape[0]-torch.sum(torch.argmax(disc_out_gen, dim=1))
      print('  t2 disc errors: {}'.format(t2_errors))

      if disc_loss > 0.5:
        set_lr(disc_optimizer, 1e-2)
      elif disc_loss > 0.3:
        set_lr(disc_optimizer, 1e-3)
      else:
        set_lr(disc_optimizer, 1e-4)

      if disc_loss > 0.3:
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

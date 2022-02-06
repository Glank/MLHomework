import os
from os.path import join, exists
import torch
from multiprocessing import Lock

DATA_DIR = 'data/mnist'
URLS = [
  ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-imgs'),
  ('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-labels'),
  ('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'test-imgs'),
  ('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'test-labels'),
]

def download(url, out_dir, fn):
  os.system('mkdir -p {}'.format(out_dir))
  full_fn = join(out_dir, fn)
  os.system('wget {} -O {}.gz'.format(url, full_fn))
  os.system('gzip -d {}.gz'.format(full_fn))

def download_all():
  global URLS, DATA_DIR
  for url, fn in URLS:
    full_fn = join(DATA_DIR, fn)
    if not exists(full_fn):
      download(url, DATA_DIR, fn)

def stream_mnist(img_fn, label_fn, start=0, end=None, dtype=torch.float, only_label=None):
  # read the images
  with open(img_fn, 'rb') as img_f:
    hdr_data = img_f.read(16)
    magic = int.from_bytes(hdr_data[:4], 'big')
    if magic != 2051:
      raise RuntimeError("Invalid magic numbr in {}".format(img_fn))
    n_imgs = int.from_bytes(hdr_data[4:8], 'big')
    rows = int.from_bytes(hdr_data[8:12], 'big')
    cols = int.from_bytes(hdr_data[12:16], 'big')

    if end is None or end > n_imgs:
      end = n_imgs
    if start >= end:
      return

    with open(label_fn, 'rb') as lbl_f:
      hdr_data = lbl_f.read(8)
      magic = int.from_bytes(hdr_data[:4], 'big')
      if magic != 2049:
        raise RuntimeError("Invalid magic numbr in {}".format(label_fn))
      n_labels = int.from_bytes(hdr_data[4:8], 'big')
      assert(n_imgs == n_labels)

      for i in range(start, end):
        img_f.seek(16+rows*cols*i)
        img_data = bytearray(rows*cols)
        img_data[:] = img_f.read(rows*cols)
        img = torch.frombuffer(img_data, dtype=torch.uint8)
        img = img.reshape((rows, cols)).type(dtype)

        lbl_f.seek(8+i)
        label = torch.zeros((10,), dtype=dtype)
        label_n = lbl_f.read(1)[0]
        label[label_n] = 1
        if only_label is None or label_n == only_label:
          yield img, label

def open_mnist(img_fn, label_fn):
  # read the images
  with open(img_fn, 'rb') as f:
    img_data = f.read()
  magic = int.from_bytes(img_data[:4], 'big')
  if magic != 2051:
    raise RuntimeError("Invalid magic numbr in {}".format(img_fn))
  n_imgs = int.from_bytes(img_data[4:8], 'big')
  rows = int.from_bytes(img_data[8:12], 'big')
  cols = int.from_bytes(img_data[12:16], 'big')
  imgs = torch.frombuffer(img_data, dtype=torch.uint8, offset=16)
  imgs = imgs.reshape((n_imgs, rows, cols))

  # read the labels
  with open(label_fn, 'rb') as f:
    label_data = f.read()
  magic = int.from_bytes(label_data[:4], 'big')
  if magic != 2049:
    raise RuntimeError("Invalid magic numbr in {}".format(label_fn))
  n_labels = int.from_bytes(label_data[4:8], 'big')
  assert(n_imgs == n_labels)
  labels = torch.frombuffer(label_data, dtype=torch.uint8, offset=8)

  return imgs, labels

def training_data():
  download_all()
  return open_mnist(join(DATA_DIR, 'train-imgs'), join(DATA_DIR, 'train-labels'))

def test_data():
  download_all()
  return open_mnist(join(DATA_DIR, 'test-imgs'), join(DATA_DIR, 'test-labels'))

def print_img(img):
  chars = list(' .;-+=O#@')
  for r in range(img.shape[0]):
    for c in range(img.shape[1]):
      print(chars[int(img[r,c]/256.*len(chars))], end='')
    print('')

class MNISTDataset(torch.utils.data.IterableDataset):
  download_lock = Lock()
  def __init__(self, start=0, end=None, test=False, only_label=None):
    super(MNISTDataset).__init__()
    assert end is None or end>=start
    self.start = start
    self.end = end
    self.test = test
    self.only_label = only_label
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      start = self.start
      end = self.end
    else:
      per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
      worker_id = worker_info.id
      start = self.start + worker_id * per_worker
      end = min(start + per_worker, self.end)
    MNISTDataset.download_lock.acquire()
    try:
      download_all()
    finally:
        MNISTDataset.download_lock.release()
    imgs_fn = join(DATA_DIR, 'test-imgs' if self.test else 'train-imgs')
    labels_fn = join(DATA_DIR, 'test-labels' if self.test else 'train-labels')
    return stream_mnist(imgs_fn, labels_fn, start, end, only_label=self.only_label)

def main():
  download_all()
  imgs, labels = test_data()
  for i in range(10):
    img = imgs[i, :, :]
    label = labels[i]
    print('Label: {}'.format(label))
    print_img(img)

def main2():
  download_all()
  imgs_fn = join(DATA_DIR, 'train-imgs')
  labels_fn = join(DATA_DIR, 'train-labels')
  for img, label in stream_mnist(imgs_fn, labels_fn, 10, 20):
    print('Label: {}'.format(label))
    print_img(img)

def main3():
  ds = MNISTDataset()
  batch_size = 64
  for X,y in torch.utils.data.DataLoader(ds, batch_size=batch_size):
    print(X.shape)
    print(y.shape)
    print('Label: {}'.format(y[0]))
    print_img(X[0,:,:])

if __name__ == '__main__':
  main3()

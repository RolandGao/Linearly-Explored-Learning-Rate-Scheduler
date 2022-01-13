mkdir pycls/datasets/data
cd pycls/datasets/data
wget curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
mv cifar-10-batches-py cifar10

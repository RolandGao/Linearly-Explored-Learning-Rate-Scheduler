mkdir pycls/datasets/data
cd pycls/datasets/data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -xzf imagenette2.tgz
rm imagenette2.tgz
mv imagenette2 imagenet

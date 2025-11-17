cd lib
#git clone https://github.com/CompVis/taming-transformers.git 
cd taming-transformers
#git checkout 3ba01b2
mkdir -p logs/vqgan_imagenet_f16_1024/checkpoints
mkdir -p logs/vqgan_imagenet_f16_1024/configs
wget 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt'
wget 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1' -O 'logs/vqgan_imagenet_f16_1024/configs/model.yaml'
cd ../../
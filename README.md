# MQBench Quantization Aware Training with PyTorch

I am using MQBench(http://mqbench.tech/) to quantize the model for deployment.

MQBench is a benchmark and framework for evluating the quantization algorithms under real world hardware deployments. 

## Prerequisites
- Python 3.7+
- PyTorch 1.8.1+

## Install MQBench Lib
Before run this repository, you should install MQBench:
```
git clone https://github.com/ModelTC/MQBench.git
cd MQBench
python setup.py build
python setup.py install
```

## Training Fp32 Model
```
# Start training fp32 model with: 
# model_name can be ResNet18, MobileNet, ...
python main.py model_name

# You can manually config the training with: 
python main.py --resume --lr=0.01
```
## Training Quantize Model
```
# Start training quantize model with: 
# model_name can be ResNet18, MobileNet, ...
python main.py model_name --quantize

# You can manually config the training with: 
python main.py --resume --parallel DP --BackendType Tensorrt --quantize
python -m torch.distributed.launch main.py --local_rank 0 --parallel DDP --resume  --BackendType Tensorrt --quantize
```


# MQBench Quantization Aware Training with PyTorch

I am using MQBench(Model Quantization Benchmark)(http://mqbench.tech/) to quantize the model for deployment.

MQBench is a benchmark and framework for evluating the quantization algorithms under real world hardware deployments. 

## Prerequisites
- Python 3.7+
- PyTorch == 1.8.1

## Install MQBench Lib
Before run this repository, you should install MQBench:

Notice that MQBench version is 0.0.2.

```
git clone https://github.com/ZLkanyo009/MQBench.git
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

## Accuracy
| Model                                                | Acc.(fp32) | Acc.(tensorRT) |
| ---------------------------------------------------- | ------ | ------ |
| [VGG16](https://arxiv.org/abs/1409.1556)             | 79.90% |78.95%|
| [GoogleNet](https://arxiv.org/abs/1409.4842)         | 90.20% |89.42%|
| [ResNet18](https://arxiv.org/abs/1512.03385)         | 95.43% |95.44%|
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)    | 89.47% |89.22%|
| [SENet18](https://arxiv.org/abs/1709.01507)          | 91.69% |91.34%|
| [MobileNetV2](https://arxiv.org/abs/1801.04381)      | 88.42% |87.65%|
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431) | 87.07% |86.95%|
| [SimpleDLA](https://arxiv.org/abs/1707.064)          | 90.24% |89.45%|
| [DenseNet121](https://arxiv.org/abs/1608.06993)      | 85.18% |85.10%|
| [PreActResNet18](https://arxiv.org/abs/1603.05027)   | 92.06% |91.68%|

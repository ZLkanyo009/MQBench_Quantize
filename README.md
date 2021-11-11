# MQBench Quantization Aware Training with PyTorch

I am using MQBench(Model Quantization Benchmark)(http://mqbench.tech/) to quantize the model for deployment.

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

## Fp32 Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

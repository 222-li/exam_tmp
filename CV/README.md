README.md
# CIFAR10 图像分类模型对比实验

## 项目简介

本项目基于 CIFAR10 数据集完成图像分类任务，对比分析三种不同网络结构的性能表现：

- ResNet18（残差网络）
- VGG19（经典卷积神经网络）
- Vision Transformer（ViT）

实验目标不仅是比较准确率，更关注不同网络结构的设计思想、模型收敛速度和错误案例分析。

---

## 项目结构

```text
project/
├── README.md
├── requirements.txt
├── src/
│   ├──pytorch-cifar-master
│   ├── vit-pytorch-main
├── results/
│   ├── resnet_curve.png
│   ├── vgg_curve.png
│   ├── error_samples│
└── report.pdf
环境配置
CPU 训练（未使用 GPU）
Python 3.10
PyTorch 2.x
torchvision 0.16.x
NumPy

数据集准备
数据集：CIFAR10
自动下载到 data/ 文件夹
数据集信息：
类别数：10
训练集：50,000
测试集：10,000
图像大小：32×32

类别：

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
数据增强：
随机裁剪（padding=4）
随机水平翻转

归一化：

transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
训练命令
训练 ResNet18
python main.py
训练 VGG19
python main.py
训练 ViT
python train_vit_decorr.py

所有训练脚本均记录前 20 个 Epoch 的 Loss 和 Accuracy，并保存错误分类图片。
ResNet和VGG模型均出自一个仓库：https://github.com/kuangliu/pytorch-cifar
VIT仓库为：https://github.com/lucidrains/vit-pytorch
测试命令

若提供独立测试脚本：python evaluate.py

如未提供，训练脚本会在每个 Epoch 自动计算测试集 Accuracy。

实验结果
模型	最高 Accuracy
ResNet18	78%
VGG19	    79%
ViT	        无(原仓库没有测试集)
ResNet18 收敛快，训练稳定
VGG19 参数量大，但准确率接近 ResNet18
ViT 捕捉全局特征能力强，训练成本较高
错误案例分析

典型误分类：

ResNet18：Cat → Dog
VGG19：Deer → Horse
ViT：Frog → Dog

原因分析：

CIFAR10 图像尺寸小（32×32），细节信息有限
类别间视觉特征相似，容易混淆
部分图像背景复杂，局部特征提取不足

改进策略：

增加数据增强（旋转、翻转、裁剪）
增加训练 Epoch
混合模型结构优化
结果文件说明

results/ 文件夹包含：

ResNet/VGG/ViT Loss & Accuracy 曲线图
错误分类图片
模型训练结果统计文件

实验总结
ResNet18 在性能和训练效率之间取得平衡
VGG19 网络较深，但性能提升有限
ViT 利用全局特征略优，但计算成本高


详细实验分析请见：report.pdf
包含：
文献综述
技术演进分析
模型对比
实验环境
实验结果
错误案例分析
调参策略与复盘总结
参考文献
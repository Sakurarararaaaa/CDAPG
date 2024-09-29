## Introduction
1.2本工作强调了通过颜色反卷积整合先验知识以推进胶质瘤诊断和预后评估的潜力。模型利用颜色反卷积提取颜色异常图，引导网络聚焦于正染色区域。模型增强了特征提取能力，提高了分类精度和可解释性。实验结果表明，与基线模型相比，CD34数据集的精度、召回率和f1得分分别提高了9.17%、9.35%和12.35%。

## Project Structure

```
├── datasets/                 # 数据集文件夹
├── networks/                 # 模型文件夹
│   ├── DIYResNet18        
│   ├── CD_MyResNet18      
├── utils/
│   └── gen_dataset_txt.py
├── train.py             	# 训练脚本
└── README.md             	# 项目说明文件

```


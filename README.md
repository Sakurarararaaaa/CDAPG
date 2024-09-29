## Introduction
Immunohistochemistry (IHC) plays a crucial role in pathological evaluation, providing accurate assessments and personalized treatment plans. CD34-stained images are used to identify tumor vascular endothelial cells. Different tumors exhibit various CD34 expression patterns, making accurate analysis of these stained regions critical for diagnosis. This study proposes a novel color deconvolution-aware prior-guided model for automating the analysis of CD34-stained glioma IHC images. The model utilizes color deconvolution to extract color abnormality maps, guiding the network to focus on positively stained regions. This method enhances feature extraction capabilities, improving classification accuracy and interpretability. Experimental results show that, compared to baseline models, Precision, Recall, and F1-score on the CD34 dataset were improved by 9.17%, 9.35%, and 12.35%, respectively. Our work emphasizes the potential of integrating prior knowledge through color deconvolution to advance glioma diagnosis and prognosis evaluation.
## Project Structure

```
├── datasets/                 # Dataset folder
├── networks/                 # Model folder
│   ├── DIYResNet18        
│   ├── CD_MyResNet18      
├── CDF/                 # Color abnormality map extraction.
│   └── gen_dataset_txt.py
├── train.py             	# Training script
└── README.md             	# 项目说明文件

```


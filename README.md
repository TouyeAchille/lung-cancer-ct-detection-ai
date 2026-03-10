# Project
Développement d’un système d’intelligence artificielle pour la détection précoce du cancer du poumon à partir d’images scanner (CT-scan)

## L’objectif 
Développer un système d’aide au diagnostic basé sur IA capable de :
• Détecter automatiquement les nodules pulmonaires à partir d’images CT-scan 
• Classifier les nodules (bénins vs suspects/malignes).
• Fournir un score de risque interprétable pour le clinicien.

Ce projet se déroule en 5 étapes principales :

1. Réaliser une recherche bibliographique sur les méthodes d’analyse d’images médicales
basées sur le Deep Learning :

- **Fine-tuning Model**
 
* **3 modèles CNN classiques : (ResNet-50, EfficientNet-B0, VGG16)**
* **1 modèle Vision Transformer (ViT)** 

Exemple PyTorch :

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes

model = model.to("cuda")

etc ...
```
---

* **EfficientNet-B0**

```python
from torchvision.models import efficientnet_b0

model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
etc ...
```
---
```
import torch.nn as nn
from torchvision import models

model = models.vgg16(pretrained=True)

model.classifier[6] = nn.Linear(4096, 4)  # 4 classes

model = model.to("cuda")
```

## Vision Transformer (modèle moderne) : **ViT-Base Patch16-224**

## Fine-tuning ViT avec HuggingFace

```python
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=4
)
```
Préprocessing :

```python
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)
```


3. Développer un pipeline de prétraitement des images médicales (normalisation, segmentation pulmonaire, augmentation de données) : pipeline complet CT-scan → Deep Learning → détection du cancer.

4. Concevoir et entraîner un modèle de détection et de classification des nodules pulmonaires,

5 Evaluer ses performances ( Precision, F1-score, etc.. ).

## Bases de données publiques
1 - <a href="https://www.cancerimagingarchive.net/collection/lidc-idri/"> cancerimagingarchive.net </a>
2 - <a href="https://www.cancerimagingarchive.net/analysis-result/dicom-lidc-idri-nodules/"> dicom-lidc-idri-nodules </a> 


## Ressources
1 - <a href="https://project-monai.github.io/">project-monai.github.io/</a> 

2 - pylidc (accès facile aux annotations) SimpleITK (lire les CT)

4 - <a href="https://www.dicomstandard.org/current/"> dicomstandard </a>

5 - <a href="https://github.com/DorsaRoh/LungAI">:LungAI </a> 

6 - <a href="https://huggingface.co/dorsar/lung-cancer-detection"> lung-cancer-detection </a> 



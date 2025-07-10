### k-Means Mask Mamba
[![CC BY 4.0][cc-by-shield]][cc-by]
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
Official Pytorch implementation of DSM, proposed in the following paper:

[Unleashing Diffusion and State Space Models for Medical Image Segmentation](https://www.arxiv.org/abs/2506.12747).  

## Installation

- It is recommended to clone the repository with Python 3.9:

  ```bash
  git clone https://github.com/Rows21/k-Means_Mask_Mamba.git
  cd k-Means_Mask_Mamba

## Datasets
- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT)
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details)
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block)
- 06 [Liver segmentation (3D-IRCADb)](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/)
- 07 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD)
- 08 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- 09 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org)
- 10 [Decathlon (Liver, Lung, Pancreas, HepaticVessel, Spleen, Colon](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
- 11 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- 12 [AbdomenCT 12organ](https://zenodo.org/records/7860267)

## Mask Generation
1. Please refer to [CLIP-Driven](https://github.com/ljwztc/CLIP-Driven-Universal-Model) to organize the downloaded datasets.
2. Modify [ORGAN_DATASET_DIR](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L51) and [NUM_WORKER](https://github.com/zongzi3zz/CAT/blob/2146b2e972d0570956c52317a75c823891a4df2c/label_transfer.py#L53) in label_transfer.py  
3. `python -W ignore label_transfer.py`

## Citation
If you find this repository helpful, please consider citing:
```
@article{wu2025unleashing,
  title={Unleashing Diffusion and State Space Models for Medical Image Segmentation},
  author={Wu, Rong and Chen, Ziqi and Zhong, Liming and Li, Heng and Shu, Hai},
  journal={arXiv preprint arXiv:2506.12747},
  year={2025}
}
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

© 2025 Rong Wu. You are free to share and adapt the material with attribution.



 
 



# k-Means Mask Mamba

<p align="center">
    <a href='[https://arxiv.org/abs/2502.18756](https://arxiv.org/abs/2506.12747)'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'>
    </a>
    <a href='https://creativecommons.org/licenses/by-nc/4.0/'>
      <img src='https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg'>
    </a>
    <a href='https://doi.org/10.5281/zenodo.15852175'>
      <img src='https://zenodo.org/badge/427764203.svg'>
    </a>
  </p>
[Unleashing Diffusion and State Space Models for Medical Image Segmentation](https://www.arxiv.org/abs/2506.12747).  

## Installation

- It is recommended to clone the repository with Python 3.9:

  ```bash
  git clone https://github.com/Rows21/k-Means_Mask_Mamba.git
  cd k-Means_Mask_Mamba

## Datasets
The data that support the findings of this study are openly available as follows:
- 01 [Multi-Atlas Labeling Beyond the Cranial Vault - Workshop and Challenge (BTCV)](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), reference [57].
- 02 [Pancreas-CT TCIA](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT), reference [69].
- 03 [Combined Healthy Abdominal Organ Segmentation (CHAOS)](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/), reference [54].
- 04 [Liver Tumor Segmentation Challenge (LiTS)](https://competitions.codalab.org/competitions/17094#learn_the_details), reference [45].
- 05 [Kidney and Kidney Tumor Segmentation (KiTS)](https://kits21.kits-challenge.org/participate#download-block), reference [50].
- 06 [WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image](https://github.com/HiLab-git/WORD), reference [61].
- 07 [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K), reference [62].
- 08 [Multi-Modality Abdominal Multi-Organ Segmentation Challenge (AMOS)](https://amos22.grand-challenge.org), reference [53].
- 09 [MSD](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2), reference [43].
- 10 [CT volumes with multiple organ segmentations (CT-ORG)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890), reference [68].
- 11 [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), reference [73].

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

## Reference
[43] Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B. A., et al. (2022). The medical segmentation decathlon. Nature Communications, 13(1), 4128.

[45] Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., et al. (2023). The liver tumor segmentation benchmark (LiTS). Medical Image Analysis, 84, 102680.

[50] Heller, N., McSweeney, S., Peterson, M. T., et al. (2020). An international challenge to use artificial intelligence to define the state-of-the-art in kidney and kidney tumor segmentation in CT imaging. J. Clin. Oncol., 38(6 Suppl), 626.

[53] Ji, Y., Bai, H., Ge, C., Yang, J., Zhu, Y., Zhang, R., et al. (2022). Amos: A large-scale abdominal multi-organ benchmark for versatile medical image segmentation. NeurIPS, 35:36722–36732.

[54] Kavur, A. E., Gezer, N. S., Barış, M., Aslan, S., Conze, P. H., Groza, V., et al. (2021). Chaos challenge-combined (ct-mr) healthy abdominal organ segmentation. Medical Image Analysis, 69, 101950.

[57] Landman, B., Xu, Z., Iglesias, J., Styner, M., Langerak, T., Klein, A. (2015). MICCAI multi-atlas labeling beyond the cranial vault–workshop and challenge. In MICCAI Challenge, vol. 5, p. 12.

[61] Luo, X., Liao, W., Xiao, J., Chen, J., Song, T., Zhang, X., et al. (2022). WORD: A large-scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image. Medical Image Analysis, 82, 102642.

[62] Ma, J., Zhang, Y., Gu, S., Zhu, C., Ge, C., Zhang, Y. (2021). AbdomenCT-1K: Is abdominal organ segmentation a solved problem? TPAMI, 44(10):6695–6714.

[68] Rister, B., Yi, D., Shivakumar, K., Nobashi, T., Rubin, D. L. (2020). CT-ORG, a new dataset for multiple organ segmentation in computed tomography. Scientific Data, 7(1):381.

[69] Roth, H. R., Lu, L., Farag, A., Shin, H. C., Liu, J., Turkbey, E. B., et al. (2015). DeepOrgan: Multi-level deep convolutional networks for automated pancreas segmentation. In MICCAI, pp. 556–564.

[73] Wasserthal, J., Breit, H.-C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., et al. (2023). TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. Radiology: Artificial Intelligence, 5(5):e230024.

## License
Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

© 2025 Rong Wu. You are free to share and adapt the material with attribution.



 
 



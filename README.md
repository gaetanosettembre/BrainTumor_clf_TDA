# Title Article

This repository is the official code for the paper "A Novel Diagnostic Decision-support System for Brain Tumor Detection and Classification in MRI Scans" by Serena Grazia De Benedictis, Grazia Gargano and Gaetano Settembre.

![Workflow](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/workflow_BT.png?raw=true)


## Citation
If you find the project codes useful for your research, please consider citing


```
@ARTICLE{,
  author={De Benedictis, Serena Grazia and Gargano, Grazia and Settembre, Gaetano},
  journal={}, 
  title={A Novel Diagnostic Decision-support System for Brain Tumor Detection and Classification in MRI Scans}, 
  year={},
  volume={},
  number={},
  pages={},
  keywords={Brain tumors, Decision-support system, Magnetic resonance imaging, Topological data analysis, Classification, Machine learning},
  doi={}
}

```

## Dependencies

The code relies on the following Python 3.9.XX + libs. Packages needed are:
 
* Ripser
* Tensorly
* OpenCV
* Numpy
* Matplotlib
* Scikit-learn

All dependencies in requirements.txt

## Installation/Usage

Create python env and install all needed dependencies:

    conda create -n "braintumor" python=3.9

    conda activate braintumor

    pip install -r requirements.txt

## Data

The dataset used is available [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (from Kaggle.com).

## Results

Classification results:
![Confusion Matrix and Classification report](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/res_cls.png?raw=true)

Examples of extracted ROIs:
![ROI_glioma](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/roi_glioma.png)
![ROI_meningioma](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/roi_meningioma.png)
![ROI_pituitary](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/roi_pituitary.png)
![ROI_notumor](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/roi_notumor.png)

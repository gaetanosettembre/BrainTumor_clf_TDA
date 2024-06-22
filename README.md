# Title Article

This repository is the official code for the paper "Enhanced MRI Brain Tumor Classification and Detection via Low-Rank Tensor Decomposition and Topological Data Analysis" by Serena Grazia De Benedictis, Grazia Gargano and Gaetano Settembre.

Link to full article [here](https://pages.github.com/).

![Workflow](https://github.com/gaetanosettembre/BrainTumor/blob/main/images/workflow_BT.png?raw=true)


## Citation
If you find the project codes useful for your research, please consider citing


```
@ARTICLE{braintumor_tda,
  author={De Benedictis, Serena Grazia and Gargano, Grazia and Settembre, Gaetano},
  journal={Journal of Computational Mathematics and Data Science}, 
  title={Enhanced MRI Brain Tumor Classification and Detection via Low-Rank Tensor Decomposition and Topological Data Analysis}, 
  year={2024},
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

## License

The entire content of this repository is released as free (as in "libre") under the [GNU GPL v3 _only_ License](LICENSE) or the [CC BY-SA 4.0 License](https://creativecommons.org/licenses/by-sa/4.0/legalcode) as follows:

- All the source code is released under the GNU GPL v3 only;
- All the content (either textual, visual or audio) is released under the CC BY-SA 4.0.

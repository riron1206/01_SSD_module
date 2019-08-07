# 01_SSD_module
SSD(Single Shot MultiBox Detector) 

## Setup
- Anaconda 4.4.10: https://www.anaconda.com/distribution/
- NVIDIA Driver for GeForce 1080: http://www.nvidia.co.jp/Download/index.aspx?lang=jp
- Visual Studio 2015 Update 3(Visual Studio Community 2015 with Update 3): https://my.visualstudio.com/Downloads  
	- Custom installation of Visual Studio 2015
- Visual C ++ Redistributable Package for Visual Studio 2015: https://www.microsoft.com/ja-JP/download/details.aspx?id=48145
- CUDA Toolkit 9.0: https://developer.nvidia.com/cuda-90-download-archive  
	- Installation execution except "Visual studio integration"
- cuDNN v7.0.5: https://developer.nvidia.com/rdp/cudnn-download
```bash
$ conda create -n tfgpu_py36_v3
$ activate tfgpu_py36_v3
$ conda install python=3.6
$ conda install -c anaconda tensorflow-gpu=1.8 
$ conda install -c conda-forge keras=2.1.5 
$ conda install -c conda-forge lightgbm=2.2.3 scikit-learn=0.20.3 opencv=4.1.0 grpcio=1.16 numba=0.38.1 pandas jupyter Cython Protobuf Pillow lxml Matplotlib tqdm future graphviz pydot pytest pyperclip networkx selenium beautifulsoup4 cssselect openpyxl pypdf2 python-docx requests tweepy textblob seaborn scikit-image imbalanced-learn colorlog sqlalchemy papermill shapely imageio git shap eli5 umap-learn plotly ipysheet bqplot rise bokeh jupyter_contrib_nbextensions yapf flask joblib xgboost alembic dill xlrd nose xlsxwriter lime
```

## License
This software is released under the MIT License, see LICENSE.

## Author
- Github: [riron1206](https://github.com/riron1206)
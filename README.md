# RICO-Benchmark and Detectron2 for Incremental Object Detection

> **This repository holds the framework and evaluation code for the ICCV Workshops 2025 Paper *"RICO: Two Realistic Benchmarks and an In-Depth Analysis for Incremental Learning in Object Detection"*.**  
>  
> [Read the paper on arXiv](https://arxiv.org/abs/2508.13878)  
>  
> The methods evaluated in the paper are not provided.  
>  
> Instructions for setting up the D-RICO and EC-RICO benchmark are given in the `rico-benchmark` folder.

## Abstract

Incremental Learning (IL) trains models sequentially on new data without full retraining, offering privacy, efficiency, and scalability. IL must balance adaptability to new data with retention of old knowledge. However, evaluations often rely on synthetic, simplified benchmarks, obscuring real-world IL performance. To address this, we introduce two **Realistic Incremental Object Detection Benchmarks** (RICO): **Domain RICO** (D-RICO) features domain shifts with a fixed class set, and **Expanding-Classes RICO** (EC-RICO) integrates new domains and classes per IL step. Built from 14 diverse datasets covering real and synthetic domains, varying conditions (e.g., weather, time of day), camera sensors, perspectives, and labeling policies, both benchmarks capture challenges absent in existing evaluations. Our experiments show that all IL methods underperform in adaptability and retention, while replaying a small amount of previous data already outperforms all methods. However, individual training on the data remains superior. We heuristically attribute this gap to weak teachers in distillation, single models’ inability to manage diverse tasks, and insufficient plasticity. D-RICO and EC-RICO will be made publicly available.

## Cite

If you use this framework or the RICO benchmarks for your publication, please cite our paper.

```tex
@inproceedings{neuwirthtrapp2025rico,
  author    = {Matthias Neuwirth{-}Trapp and Maarten Bieshaar and
               Danda Pani Paudel and Luc Van Gool},
  title     = {{RICO}: Two Realistic Benchmarks and an In-Depth Analysis
               for Incremental Learning in Object Detection},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  year      = {2025}
}
```

## Installing

1. Remove all saved pip data

   ```bash
   python3 -m pip cache purge
   ```

2. Install conda, cuda 11.6.2, cudnn 11.6_v8.4 and gcc 9.2.0

3. Create and activate a conda environment

   ```bash
   conda create --name det2IL python=3.8 -y
   conda activate det2IL
   ```

5. Install PyTorch and requirements. It is important to not use a different pytorch and CUDA version.

   ```bash
   pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
   pip install -r requirements.txt
   ```

6. Install Apex

   Clone the repository

   ```bash
   git clone https://github.com/NVIDIA/apex
   cd apex
   ```

   Install it. The correct CUDA version must be running. Use the same CUDA version as you used for PyTorch.

   ```bash
   # if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
   # otherwise
   pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   ```

7. Install xFormers

   The newest version of xFormers does not support the above installed PyTorch version. Therefore, an older version needs to be installed. We used commit `7e05e2c`, i.e., changed the branch in the install command from `main ` to `7e05e2c`. It is also important to compile it on the same hardware you use for running the network. This means, you should use this command:

   ```bash
   pip install -v -U git+https://github.com/facebookresearch/xformers.git@7e05e2c#egg=xformers
   ```

8. Install MMCV

   ```bash
   pip install -U openmim
   mim install mmcv
   ```

9. Then, build Detectron2-IL from source:

   ```bash
   cd /path/to/detectron2-IL
   python -m pip install -e .
   ```

10. Some specific package versions are needed:

    ```bash
    pip install pandas==1.3.5
    pip install fairscale==0.3.9
    pip install matplotlib==3.5.3
    pip install numpy==1.19.5
    pip install triton==2.1.0
    pip install Shapely==1.8.5.post1
    ```

    

#### Troubleshooting 

* We often had CUDA not loaded or the wrong CUDA version loaded. Therefore, if there is a CUDA-related error, always check first if CUDA 11.6 is loaded.
* We cloned the EVA-02 and Detectron2 repository on 24.07.2024; therefore, if there are errors, use the last commit available on that date.
* For xFormer, we used commit 7e05e2cas. It was the last version available that supports PyTorch 1.13, so we used the repository's state at that version. If there is an error, maybe do the same for Apex and MMCV.
* If you run the program and get an error message that `LINEAR` is not present in PIL, you can either downgrade PIL to a working version or change `LINEAR` to `BILINEAR` in the detectron2 code. I did the latter, and it works fine.



## Run Experiments

Detectron2 is config file-based, and we chose to use the `LazyConfig` style.

To run an experiment, first set up the appropriate config file. Go to `projects/continual_learning`. The configs used in the paper are located in the methods folder. To keep changes minimal, there are parent configs from which they are based on (see the `common` folder).

Take inspiration from the given configs or reuse them. To start the experiment, you must run the `cl_train.py` file in `detectron2/continual_learning`. In the call, additional arguments can be set, and overrides to the configuration can be made. See Detectron2 documentation for more details on that.

To be able to run the experiments, you need to tell Detectron2 where the dataset files are located. For this, simply set the environment variable:

```bash
export DETECTRON2_DATASETS=/path/to/datasets
```

A folder named d_rico and ec_rico should hold all the JSON files.

> [!NOTE]
>
> During development, we did not name the benchmarks D-RICO and EC-RICO, so you do not find those words in the code. It is mostly cl_multi_ad for D-RICO and cl_multi_ad_ccl for EC-RICO. We originally planned to do class continual learning, so you find ccl in many places. In the final version, this always represents the EC-RICO benchmark.



## Add a new IL method

**How to add a new method:**

1. Create a new folder in detectron2/continual_learning/methods with the name of the method, lets say `abc`

2. Copy the `train_loop.py` from `naive` to it

3. Create an `__init__.py` file and write in it the import to the train_loop and all other files that  will be created

   ```python
   from .train_loop import Trainer`
   ```
   
4. Go to the `__init__.py` from the methods folder and import there your new folder, e.g. 
   ```python
   from . import abc` 
   ```
   
   (use the name of the newly generated folder, if there are already names, just append it)

5. Go to the `__init__.py` from the continual_learning folder and add your method there as well

   ```python
   from .methods import abc`
   ```
   
6. To make it avaiable to import add it to the enviroment

   ```python
   sys.modules["detectron2.continual_learning.abc"] = abc`
   ```
   
7. Every new file, function and class your create in the folder of your method need to be added to the  `__init__.py` of your folder (see step 3)

8. Than you can import your  classes and functions from anywhere with
   ```python
   from detectron2.continual_learning.abc import Trainer
   ```
   

**Best practices for adding a new method:**

1. Follow the steps above, embed it into the framework

2. If you need to change a class, inherit it and overwrite methods or add new ones. But do not copy the classes.

   1. You need to give it a new name, but can rename it in the `__init__.py` file.

   2. Example: You want to change the `GeneralicedRCNN` class.

      1. Go to the method folder of your new method.

      2. Create a new file called `rcnn.py`

      3. Import the old `GeneralicedRCNN` by 

         ```python
         from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN`
         ```
         
      4. Create a new class and inherit the old class

         ```python
         class GeneralizedRCNNABC(GeneralizedRCNN)`
         ```
         
      5. If you want to change the forward path, create a new method in the class called `forward`.

         You have all the functionality of the old `GeneralizedRCNN` but overwrite the forward path with your new one.

      6. To be able to use this new  class, we add it to the `__init__.py` in the method folder like this

         ```python
         from .rcnn import GeneralizedRCNNABR as GeneralizedRCNN
         ```

      7. By importing it `as` using the old name, we create a consistent implementation

      8. In the config file, you can import this new class and replace it with the old one
      
         ```python
         from detectron2.continual_learning.abc import GeneralizedRCNN
         model._target_ = GeneralizedRCNN
         ```
      
      9. Repeat this for all the changes you do. This way code replication is minimized and everything is consistent.

## License

This repository is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.

For a list of other open source components included in PROJECT-NAME, see the file [3rd-party-licenses.txt](3rd-party-licenses.txt).

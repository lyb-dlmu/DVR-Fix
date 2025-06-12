# DVR-Fix
## DVR-Fix: A Multi-Stage Denoising Approach for Noise-Robust Vulnerability Repair
Python library dependencies:  
torch -v:1.10.2+cu113  
numpy -v:1.22.3  
tqdm -v:4.62.3  
pandas -v:1.4.1  
datasets -v:2.0.0  
gdown -v:4.5.1  
scikit-learn -v:1.1.2  
tree-sitter -v:0.20.0  
argparse -v:1.4.0  

Dataset:  
Download necessary data and unzip via the following command:  
```bash
cd data
sh download_data.sh 
cd ..
```
First of all, clone this repository to your local machine and access the main dir via the following command:
```bash
git clone https://github.com/awsm-research/VQM.git
cd DVR-Fix
```
Then, install the python dependencies via the following command:
  
```bash
cd DVR-Fix
pip install -r requirements.txt
cd DVR-Fix/transformers
pip install .
cd ../..
```
We highly recommend you check out this installation guide for the "torch" library so you can install the appropriate version on your device.

To utilize GPU (optional), you also need to install the CUDA library. You may want to check out this installation guide.

Python 3.9.7 is recommended, which has been fully tested without issues.




### Reproduction of Experiments
#### Reproduce Section 4 - RQ1

- **DVR-Fix(Proposed Approach)**

    - **Retrain Localization Model**
        ```bash
        cd DVR-Fix
        sh run_pretrain_loc.sh
        sh run_train_loc.sh
        cd ..
        ```

    - **Retrain Repair Model**
       ```bash
       cd DVR-Fix
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ..
       ```
 - **VQM**

    - **Inference**
        ```bash
        cd VQM/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ..
        ```

    - **Retrain Localization Model**
       ```bash
       cd VQM
       sh run_pretrain_loc.sh
       sh run_train_loc.sh
       cd ..
       ```
    - **Retrain Repair Model**
       ```bash
       cd VQM
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ..
       ```
- **VulRepair**

    - **Inference**
        ```bash
        cd baselines/VulRepair/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/VulRepair
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
 - **TFix**

    - **Inference**
        ```bash
        cd baselines/TFix/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/TFix
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **GraphCodeBERT**

    - **Inference**
        ```bash
        cd baselines/GraphCodeBERT/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/GraphCodeBERT
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **CodeBERT**

    - **Inference**
        ```bash
        cd baselines/CodeBERT/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/CodeBERT
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
 - **VRepair**

    - **Inference**
        ```bash
        cd baselines/VRepair/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/VRepair
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```
  - **SequenceR**

    - **Inference**
        ```bash
        cd baselines/SequenceR/saved_models/checkpoint-best-loss
        sh download_models.sh
        cd ../..
        sh run_test.sh
        cd ../..
        ```

    - **Retrain**
       ```bash
       cd baselines/SequenceR
       sh run_pretrain.sh
       sh run_train.sh
       sh run_test.sh
       cd ../..
       ```


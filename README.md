# DSE-HNGCN

#####                      — predicting the frequencies of drug-side effects based on heterogeneous networks with mining interactions between drugs and side effects.

<div style ="text-align:justify">Evaluating the frequencies of drug-side effects is crucial in drug development and risk-benefit analysis. While existing deep learning methods show promise, they have yet to explore using heterogeneous networks to simultaneously model the various relationship between drugs and side effects, highlighting areas for potential enhancement. In this study, we propose DSE-HNGCN, a novel method that leverages heterogeneous networks to simultaneously model the various relationships between drugs and side effects. By employing multi-layer graph convolutional networks, we aim to mine the interactions between drugs and side effects to predict the frequencies of drug-side effects. To address the over-smoothing problem in graph convolutional networks and capture diverse semantic information from different layers, we introduce a layer importance combination strategy. Additionally, we have developed an integrated prediction module that effectively utilizes drug and side effect features from different networks. Our experimental results, using benchmark datasets in a range of scenarios, show that our model outperforms existing methods in predicting the frequencies of drug-side effects. Comparative experiments and visual analysis highlight the substantial benefits of incorporating heterogeneous networks and other pertinent modules, thus improving the accuracy of DSE-HNGCN predictions. We also provide interpretability for DSE-HNGCN, indicating that the extracted features are potentially biologically significant. Case studies validate our model’s capability to identify potential side effects of drugs, offering valuable insights for subsequent biological validation experiments.</div>

##### This source code was tested on the basic environment with `conda==23.3.1` and `cuda==12.2`

## Environment Reproduce

- In order to get DSE-HNGCN, you need to clone this repo:

  ```
  git clone git@github.com:lennylv/DSE-HNGCN.git
  cd DSE-HNGCN
  ```

- Unzip the "data.zip" and "method.zip" files into the current directory, and create environment using files provided in the current directory

  ```
  unzip data.zip
  unzip method.zip
  conda env create -f environment.yml
  pip install -r requirements.txt
  ```

## Run for Train

- 1. Please use the following command to switch to the methods folder.
   ```bash
   cd method
   ```
2. For frequency prediction tasks in the warm-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_regress.py
   ```
   For association prediction tasks in the warm-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_classify.py
   ```
3. For frequency prediction tasks in the cold-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_regress_cold_start_drug.py
   ```
   For association prediction tasks in the cold-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_classify_cold_start_drug.py
   ```

## Run for Test

- 1. Please use the following command to switch to the methods folder.
   ```bash
   cd method
   ```
2. For frequency prediction tasks in the warm-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_regress_test.py
   ```
   For association prediction tasks in the warm-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_classify_test.py
   ```
3. For frequency prediction tasks in the cold-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_regress_cold_start_drug_test.py
   ```
   For association prediction tasks in the cold-start scenario, please run the following command to execute the program.
   ```bash
   python main_only_classify_cold_start_drug_test.py
   ```

#### contact

Xuhao Ma : 20225227081@stu.suda.edu.cn 

Tingfang Wu: tfwu@suda.edu.cn

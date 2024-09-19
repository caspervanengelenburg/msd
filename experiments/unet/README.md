# U-Net for Floor Plan Generation

Floor plan generation is based on architectural inputs like 'struct_in' (building boundary) and 'graph_in' (desired room layout and connections). The Segment-Anything Model (SAM) enhances the features in 'struct_in', while integrating Graph Neural Networks (GNN) into the U-Net architecture incorporates the structural information from 'graph_in'. This combination improves the U-Net’s ability to generate complex floor plans.

## 0. Prerequisites

Create a Conda environment:

```bash
conda create -n msd-unet python==3.9.0
conda activate msd-unet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 1. Pre-process Dataset

  1. Download the raw MSD dataset and locate it under './dataset' folder.
  
  ```bash
  dataset/
  ├── train/
  │   ├── struct_in/
  │   ├── graph_in/
  │   ├── struct_out/
  │   └── graph_out/
  └── test/
      ├── struct_in/
      ├── graph_in/
      ├── struct_out/
      └── graph_out/
  ```

  2. Download the SAM ViT-H model here and place it in the './data' folder.

  3. Run SAM preprocessing:

  ```bash
  cd data
  python sam_preprocess.py
  ```

  The processed dataset will be saved in './dataset_processed'.

  You can download the preprocessed dataset from the link below and place it in the './dataset_processed' folder:

  [Download Preprocessed Dataset](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/Eb06YLy1LhNOrreYD5wB0JsBMaJNUVqURGDsGdQW1UCszA?e=3TIdbY)

## 2. Train U-Net

  To train the U-Net model with the preprocessed data:

  ```bash
  python train_unet.py
  ```

  You can download the best model's weights from the link below and place it in the './checkpoints' folder:

  [Model Weights](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/Efhej0jvvUNCrbuA0-BbSNQBAgOreBt50blXik9dcwr7hw?e=fhfYlP)

## 3. Test U-Net

  Evaluate the model's performance using the IoU metric:

   ```bash
   cd evaluation
   python test_unet.py
   ```

## 4. Inference & Visualization

  Use the 'vis_unet.ipynb' notebook to visualize the generated floor plans compared to the ground truth.

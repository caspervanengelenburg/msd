# MSD: A Benchmark Dataset for Floor Plan Generation of Building Complexes

- Add tldr;
- Add link to paper, project page, data (both Kaggle and 4TU.ResearchData);

## Modified swiss dwellings datasets

- Add introduction to dataset;
- Add text about the data formats in MSD (explain structure of Kaggle etc.);
- Add links to where the data can be downloaded;
- Add links to Jupyter notebook explaining how to load and use the data.

## Floor plan generation

- Quick recap on two approaches (including results figure);
- Setup environment;
- How to train on MHD;
- How to train on U-Net.
  1. **Pre-process Dataset**

     We utilize **Segment-Anything (SAM)** to preprocess the boundary images for enchancing the image features.
     To pre-process the dataset, run below script:

     ```bash
     python sam_preprocess.py
     ```

     Below is the link to the results of preprocessed dataset:

     [Download Preprocessed Dataset](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/EUcMhruwEulHodMYw43PLhgBmsQIarxVwACzxpM9oNm0Fg?e=4bDucx)
     
  2. **Train U-Net**

     To train the U-Net model, run the `train_unet.py` script. This script will handle the model training, including loading the preprocessed data and performing validation:

     ```bash
     python train_unet.py
     ```

     We provide our best model's weights : [model_weight](https://o365skku-my.sharepoint.com/:u:/g/personal/jyt0131_o365_skku_edu/Efhej0jvvUNCrbuA0-BbSNQBAgOreBt50blXik9dcwr7hw?e=J7s77E)

  3. **Inference/Visualization**

     After training, use the `vis_unet.ipynb` Jupyter notebook to visualize the model's predictions. This notebook will load the trained model and compare the predicted floor plans with the ground truth.


  4. **Test the U-Net Model**

     To test the model on unseen data, run the `test_unet.py` script. This will evaluate the model on the test dataset and provide performance on IoU metric:

     ```bash
     python test_unet.py
     ```


## References

- Add references (if needed);
- Add acknowledgements (if need).

# Neural Representations in Hybrid Recommender Systems: Prediction versus Regularization

This is our official implementation for the paper:

Ramin Raziperchikolaei, Tianyu Li, and Young-joo Chung. 2021. Neural Representations in Hybrid Recommender Systems:
Prediction versus Regularization. In Proceedings of the 44th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’21), July 11–15, 2021, Virtual Event, Canada. ACM, Online, 5 pages.
https://doi.org/10.1145/3404835.3463051

The longer version (arXiv) can be found here: https://arxiv.org/pdf/2010.06070.pdf

In this work, we define the neural representation for prediction (NRP) framework and apply it to the autoencoder-based recommendation systems. We theoretically analyze how our objective function is related to the previous MF and autoencoder-based methods and explain what it means to use neural representations as the regularizer. We also ap- ply the NRP framework to a direct neural network structure which predicts the ratings without reconstructing the user and item in- formation. We conduct extensive experiments which confirm that neural representations are better for prediction than regularization and show that the NRP framework outperforms the state-of-the-art methods in the prediction task, with less training time and memory.

## Environment Settings
We use Keras with tensorflow as the backend.
- Python 3.6.9
- tensorflow-gpu version:  '2.1.0'

## Example to run the codes.

```
python main.py --ds ml100k --num_epochs 10 --batch_size 32 --reg 0 --lr 0.001 --decay 0.005
```


### Dataset
We provide the files to run our method on the MovieLens 100k dataset: https://grouplens.org/datasets/movielens/100k/

We also provide instructions on how to create files for any other dataset.

We have put the following files in the ml100k folder:

**trainset.npy**
- This is an N*3 numpy matrix, where N is number of ratings in the training set.
- Each row is a training instance: `userID itemID rating`

**testset.npy**
- Same format as the trainset.npy, but used as the test set.

**valset.npy**
- Same format as trainset.npy, but used as the validation set.

**item_dict.pkl**
- This is a dictionary of items' side information.
- We converted all the (categorical) features into binary features(./ml100k/dataset.py contains the details for ml100k dataset).
- The key of the dictionary is an item_id.
- The value for each key is a list of indices, where each index corresponds to the index of the binary feature vector of the item.

**user_dict.pkl**
- This is a dictionary of users' side information.
- We converted all the features to binary (./ml100k/dataset.py contains the details for ml100k dataset).
- The key of the dictionary is a user_id.
- The value fo each key is a list indices, where each index corresponds to the index of the binary feature vector of the user.

### How to run the code for a new dataset
- Create a folder with the name "new_dataset".
- Create these four files and put them in this folder: trainset.npy, testset.npy, valset.npy, item_dict.pkl, item_dict.pkl
- Run the code with the: --ds "new_dataset"
- The dataset.py file in the ml100k folder shows how to create the above five files.

### Results
We report RMSE after each epoch.
Each epoch takes around 1.5 minute.
We got RMSE=0.898 on the test set for ml100k dataset.


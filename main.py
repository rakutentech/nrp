from utils import make_model,rmse_err
import numpy as np
import math
import time
from tensorflow.keras.models import load_model
import pickle
from scipy import sparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse

parser = argparse.ArgumentParser(description='Neural Representation for Prediction (NRP)')
parser.add_argument('--ds', default='ml100k', type=str)
parser.add_argument('--lr', default=0.001, type=float, help='initial lr')
parser.add_argument('--decay', default=0.005, type=float, help='decay')
parser.add_argument('--reg', default=0, type=float, help='regularization')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--num_epochs', default=10, type=int)
args = parser.parse_args()

############## Loading the data##############
trainset = np.load('./' + args.ds + '/trainset.npy')
valset = np.load('./'+ args.ds + '/valset.npy')
testset = np.load('./'+ args.ds +'/testset.npy')

pkl_file = open('./'+ args.ds + '/item_dict.pkl', 'rb')
f = pickle.load(pkl_file)
item_dict = f
pkl_file.close()

pkl_file = open('./' + args.ds + '/user_dict.pkl', 'rb')
f = pickle.load(pkl_file)
user_dict = f
pkl_file.close()

iside_size = 0 #size of the item side information
for ll in item_dict.values():
    iside_size = max(iside_size,max(ll))

uside_size = 0  #size of the user side information
for ll in user_dict.values():
    uside_size = max(uside_size,max(ll))
iside_size +=1
uside_size +=1
############## Loading the data##############

############## Extracting the input (rating) vector input for the users and items based on the training set ##############
# We create a sparse training matrix for the training set
row_ind = trainset[:,0]
col_ind = trainset[:,1]
data = trainset[:,2]
rating_mat = sparse.coo_matrix((data, (row_ind, col_ind)))
num_user,num_item = rating_mat.shape

# create two dictionaries, one for the users, one for the items

Ru_dict={} # user dictionary. It maps each user to the index of the items, rated by that user (in the training set)
Ri_dict={} # item dictinary, it maps each item to the index of the users, who rated that item (in the training set)

for i in range(trainset.shape[0]):
    if(trainset[i,0] not in Ru_dict):
        Ru_dict[trainset[i,0]] = [[],[]]
    if (trainset[i, 1] not in Ri_dict):
        Ri_dict[trainset[i, 1]] = [[],[]]

    Ru_dict[row_ind[i]][0].append(col_ind[i])
    Ru_dict[row_ind[i]][1].append(trainset[i, 2])

    Ri_dict[col_ind[i]][0].append(row_ind[i])
    Ri_dict[col_ind[i]][1].append(trainset[i, 2])

num_user = len(user_dict) + 1
num_item = len(item_dict) + 1
############## Extracting the input (rating) vector input for the users and items based on the training set ##############

############## Making the model ##############
# create the model
model = make_model(num_user,num_item,uside_size,iside_size,args.reg,args.lr,args.decay)

batch_size = args.batch_size
num_epochs = args.num_epochs

train_size = len(trainset)
ll = math.ceil(len(trainset) / batch_size)
val_err = []

mu = np.mean(trainset[:,2])

rmse = rmse_err(model,valset,mu, uside_size, iside_size, user_dict,
                    item_dict, num_user, num_item, Ru_dict, Ri_dict)
val_err.append((rmse))
print('Initial RMSE on validation set:', rmse)
np_val = np.array(val_err)
if (np.argmin(np_val[:]) == (len(val_err) - 1)):
    model.save('nrp_model.h5')

############## Main loop of training the model ##############
for epoch in range(num_epochs):

    # shuffle the data at each epoch
    shuffle_indices = np.random.permutation(train_size)
    shuffled_data = trainset[shuffle_indices,:]

    start_time = time.time()

    # Training loop for each batch
    for batch_num in range(ll):

        # get a batch of data
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, train_size)
        data_train = shuffled_data[start_index:end_index,:]

        uid, iid, y_batch = zip(*data_train)

        # Extract user and item input features for the user_ids and item_ids in this batch
        tsize = len(uid) #number of points in the batch
        umat = np.zeros((tsize,uside_size)) # user side info for the batch
        imat = np.zeros((tsize,iside_size)) # item side info for the batch
        Rumat = np.zeros((tsize, num_item)) # user ratings in the batch
        Rimat = np.zeros((tsize, num_user)) # item ratings in the batch

        #extract the user and item input features from the dictionaries
        for i in range(tsize):
            uidx = user_dict[uid[i]]
            umat[i, uidx] = 1 # user side info
            if(uid[i] in Ru_dict):
                Rumat[i,Ru_dict[uid[i]][0]] = Ru_dict[uid[i]][1] #user ratings

            iidx = item_dict[iid[i]]
            imat[i, iidx] = 1 #item side info
            if(iid[i] in Ri_dict):
                Rimat[i,Ri_dict[iid[i]][0]] =  Ri_dict[iid[i]][1] #item rating

        y_batch = np.array(y_batch)
        y_batch = y_batch.reshape(-1,1)
        y_batch = y_batch - mu
        model.fit([Rimat,Rumat,imat,umat], [y_batch], epochs=1,verbose=0)

    # We computer the RMSE at the end of each batch
    rmse = rmse_err(model,valset, mu, uside_size, iside_size, user_dict,
                        item_dict, num_user, num_item, Ru_dict, Ri_dict)
    val_err.append((rmse))
    print('epoch#',epoch+1,' RMSE on validation set:', rmse)
    # save the model if RMSE decreased
    np_val = np.array(val_err)
    if (np.argmin(np_val[:]) == (len(val_err) - 1)):
        model.save('nrp_model.h5')
    print("--- %s seconds ---" % (time.time() - start_time))

#load the best model
model = load_model('nrp_model.h5')

#compute RMSE on the test set
rmse = rmse_err(model,testset,mu, uside_size, iside_size, user_dict,
                item_dict, num_user, num_item, Ru_dict, Ri_dict)

print('Test RMSE:',rmse)

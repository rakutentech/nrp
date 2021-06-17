import numpy as np
import math
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2


'''
make_model defines the neural network model

args:
    num_user: total number of users
    num_item: total number of items
    uside_size: size of the feature vector of the user side information
    iside_size: size of the feature vector of the item side information
    regval: regularization value of the neural network layers
    lr: learning rate in the optimizer
    decay: decay value in the optimizer
'''
def make_model(num_user,num_item,uside_size,iside_size,regval,lr,decay):
    inputs_Ri = Input(shape=(num_user,))

    x = Dense(100, activation='selu',kernel_regularizer=l2(regval))(inputs_Ri)
    x = Dense(100, activation='selu', kernel_regularizer=l2(regval))(x)
    o1 = Dense(100, activation='selu',kernel_regularizer=l2(regval))(x)

    inputs_Ru = Input(shape=(num_item,))
    x = Dense(100, activation='selu',kernel_regularizer=l2(regval))(inputs_Ru)
    x = Dense(100, activation='selu', kernel_regularizer=l2(regval))(x)
    o2 = Dense(100, activation='selu',kernel_regularizer=l2(regval))(x)

    inputs_si = Input(shape=(iside_size,))
    x = Dense(100, activation='selu',kernel_regularizer=l2(regval))(inputs_si)
    x = Dense(100, activation='selu', kernel_regularizer=l2(regval))(x)
    o3 = Dense(100, activation='selu',kernel_regularizer=l2(regval))(x)

    inputs_su = Input(shape=(uside_size,))
    x = Dense(100, activation='selu',kernel_regularizer=l2(regval))(inputs_su)
    x = Dense(100, activation='selu', kernel_regularizer=l2(regval))(x)
    o4 = Dense(100, activation='selu',kernel_regularizer=l2(regval))(x)

    concat = concatenate(([o1, o2, o3, o4])) #concatenate the representations from different sources
    x = Dense(500, activation='selu',kernel_regularizer=l2(regval))(concat)
    x = Dense(200, activation='selu',kernel_regularizer=l2(regval))(x)
    x = Dense(100, activation='selu',kernel_regularizer=l2(regval))(x)
    x = Dense(50, activation='selu',kernel_regularizer=l2(regval))(x)
    output_r = Dense(1, activation='linear')(x)

    model = Model(inputs=[inputs_Ri, inputs_Ru, inputs_si, inputs_su],
                  outputs=[output_r])

    print(model.summary())


    rms = optimizers.RMSprop(decay=decay,learning_rate=lr)
    model.compile(optimizer=rms,loss='mean_squared_error', metrics=['mae'])
    return model


'''
rmse_err computes the RMSE of the model on a set of ratings

args:
    model: the neural network model, which is defined in the make_model function above.
    valset: a matrix of size n*3, where the first/second/third columns are user_id/item_id/ratings.
            The RMSE is computed on the ratings of this matrix
    mu: average value of all the ratings in the trainnig set
    uside_size: size of the feature vector of the user side information
    iside_size: size of the feature vector of the item side information
    user_dict: a dictionary of users' side information. The key is a user_id and the value is a list of indices,
                where each index corresponds to the index of the binary feature vector of the user.
    item_dict: a dictionary of items' side information. The key is an item_id and the value is a list of indices,
                where each index corresponds to the index of the binary feature vector of the item.
    num_user: total number of users
    num_item: total number of items
    Ru_dict: a dictionary of users' interaction vectors. The key is a user_id. The value for each key is a list of [l1,l2].
                l1 is the lits of the items rated by the user, and l2 is the corresponding rating values.
    Ri_dict: a dictionary of items' interaction vectors. The key is an item_id. The value for each key is a list of [l1,l2].
                l1 is the lits of the users who rated that item user, and l2 is the corresponding rating values.
'''

def rmse_err(model,valset,mu,uside_size,iside_size,
             user_dict,item_dict,num_user,
             num_item,Ru_dict,Ri_dict):

    batch_size = 1000
    val_size = len(valset)
    val_ll = math.ceil(len(valset) / batch_size)
    err_mse = 0
    num_pairs = 0

    for batch_num in range(val_ll):

        # get a batch of data
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, val_size)
        data_train = valset[start_index:end_index, :]

        uid, iid, y_batch = zip(*data_train)

        # Extract user and item's input features for the bacth
        tsize = len(uid)
        umat = np.zeros((tsize, uside_size))
        imat = np.zeros((tsize, iside_size))
        Rumat = np.zeros((tsize, num_item))
        Rimat = np.zeros((tsize, num_user))

        for i in range(tsize):
            uidx = user_dict[uid[i]]
            umat[i, uidx] = 1
            if(uid[i] in Ru_dict):
                Rumat[i,Ru_dict[uid[i]][0]] = Ru_dict[uid[i]][1]

            iidx = item_dict[iid[i]]
            imat[i, iidx] = 1
            if(iid[i] in Ri_dict):
                Rimat[i,Ri_dict[iid[i]][0]] =  Ri_dict[iid[i]][1]

        y_batch = np.array(y_batch)
        y_batch = y_batch.reshape(-1, 1)
        y_batch = y_batch
        # compute the RMSE
        pR = model.predict([Rimat, Rumat, imat, umat]) #
        pR = pR + mu
        diff = (pR - y_batch)
        res_mse = np.sum(diff ** 2)
        err_mse = err_mse + res_mse
        num_pairs = num_pairs + tsize


    err_mse = np.sqrt(err_mse / num_pairs)

    return err_mse
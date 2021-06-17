import pandas as pd
import re
import numpy as np
import pickle

# gets a string and cleans it!
def clean_str(string):
    string = re.sub(r"\'s", " ", string)
    string = re.sub(r"\'ve", " ", string)
    string = re.sub(r"n\'t", " ", string)
    string = re.sub(r"\'re", " ", string)
    string = re.sub(r"\'d", " ", string)
    string = re.sub(r"\'ll", " ", string)
    string = re.sub(r",", " ", string)
    string = re.sub(r"!", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\.", " ", string)
    string = re.sub(r"\*", " ", string)
    string = re.sub(r"\&", " ", string)
    string = re.sub(r"\:", " ", string)

    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

#Reading users file:
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./u.user', sep='|', names=u_cols,encoding='latin-1')

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('./u.item', sep='|', names=i_cols,
encoding='latin-1')

# Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
trainset = pd.read_csv('./u.data', sep='\t', names=r_cols,encoding='latin-1')

#clean the title of each item
for i in range(items.shape[0]):
    str = items['movie title'][i] #get the title
    str = clean_str(str) #clean the string
    items.at[i,'movie title']=str #set it back!

#get the list of all the UNIQUE words
iwordlist = list(items['movie title'].str.split(' ', expand=True).stack().unique())

# word_vocab[x] returns the index of word x
item_vocab = {x: i for i, x in enumerate(iwordlist)}# assign index to each word.

#length of the vocab
iside_size = len(iwordlist) + 19 # we have 19 genres
starts_idx = len(iwordlist)

#features for the items
# item_dict{movieid} returns all the nonzero indices for the movieid in the vocab
item_dict={}

#Two for loops: one over all the items in the table, one over all the words in each item.
for i in range(items.shape[0]):
    #get the title
    words = (items['movie title'][i]).split(' ')
    # movie_id
    mid = int(items['movie id'][i])
    item_dict[mid] = []
    for word in words:
        item_dict[mid].append(item_vocab[word])

    genres = items.iloc[i,-18:] #take the last 18 columns
    genres = genres.values #convert it to numpy array
    idxs = genres.nonzero()[0] + starts_idx #get the nonzero indices as numpy array
    if(len(idxs)==0): #check if it contains any non-zero index
        continue
    lidxs = idxs.tolist() #convert to list
    item_dict[mid].extend(lidxs) #add it to the end of the other list


# we find the side information for all the users
# get all the side info
users_side = users.iloc[:,1:]
#unique side info
u_list = (pd.unique(users_side.values.flatten())).tolist()
user_vocab = {x: i for i, x in enumerate(u_list)} # assign index to each unique user side info
uside_size = len(u_list)

# maps each user id to the indices of the side info
user_dict = {}
for i in range(users.shape[0]):
    uid = int(users['user_id'][i])
    user_dict[uid] = []
    vals = ((users.iloc[i, 1:]).values.flatten()).tolist()
    for val in vals:
        user_dict[uid].append(user_vocab[val])

# trainset=[ user_id, item_id, rating], it contains all the ratings
trainset = trainset.iloc[:,0:3].values

# randomly select 80% as the training, 10% as test, and 10% as the validation
tr_size = trainset.shape[0]
shidx = np.random.permutation(tr_size)

st_idx1 = int(tr_size*.8)
st_idx2 = int(tr_size*.9)

testset = trainset[shidx[st_idx2:],:]
valset = trainset[shidx[st_idx1:st_idx2],:]
trainset=trainset[shidx[0:st_idx1],:]

#trainset, testset, valset, Ri_dict, Ru_dict, item_dict, user_dict

np.save('./trainset',trainset)
np.save('./testset', testset)
np.save('./valset', valset)

fName = open("./item_dict.pkl", "wb")
pickle.dump(item_dict, fName)
fName.close()

fName = open("./user_dict.pkl", "wb")
pickle.dump(user_dict, fName)
fName.close()




#, iside_size, uside_size, num_user, num_item




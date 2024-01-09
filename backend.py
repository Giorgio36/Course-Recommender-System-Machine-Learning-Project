import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "Neural Network")
        #   "KNN",
        #   "NMF",
        #   "Regression with Embedding Features",
        #   "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_course_genre():
    return pd.read_csv("course_genre.csv")

def load_user_profile():
    return pd.read_csv("user_profile.csv")

def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id
    
def add_new_user_profile(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        profile_df = load_user_profile()
        new_id = profile_df['user'].max() + 1
        res_dict['user'] = new_id

        for column in profile_df.columns[1:]:
            res_dict[column] = 1
        
        new_df = pd.DataFrame(res_dict, index = [len(profile_df['user'])])
        updated_profiles = pd.concat([profile_df, new_df])

        theme1 = ['database']
        theme2 = ['python']
        theme3 = ['cloud', 'computing']
        theme4 = ['data','analysis']
        theme5 = ['containers']
        theme6 = ['machine','learning']
        theme7 = ['vision','computer']
        theme8 = ['data', 'science']
        theme9 = ['big', 'data']
        theme10 = ['chatbot', 'cahtbots']
        theme11 = ['r']
        theme12 = ['backend']
        theme13 = ['front','frontend']
        theme14 = ['blockchain']

        themes = [theme1,theme2,theme3,theme4,theme5,theme6,theme7,theme8,theme9,theme10,theme11,theme12,theme13,theme14]

        bow_df = load_bow()
        for course in new_courses:
            course_bow = bow_df[bow_df['doc_id'] == course]
            words = course_bow['token'].to_list()
            for i,theme in enumerate(themes):
                check =  all(item in words for item in theme)
                if check:
                    updated_profiles[-1:].iloc[0,i+1] += 5

        updated_profiles.to_csv("user_profile.csv", index=False)

        ratings_df = load_ratings()
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)

        return new_id


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict

def combine_cluster_labels(labels, user_ids):
  labels_df  = pd.DataFrame(labels)
  cluster_df = pd.merge(user_ids, labels_df, left_index = True, right_index = True)
  cluster_df.columns = ['user','cluster']
  return cluster_df

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def user_profile_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, course_genres_df, user_vector ):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {}
    # First find all enrolled courses for user
    unselected_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unselected_course_ids)]
    unselected_course_matrix = unselected_course_df.iloc[:,2:].values
    unselected_course_ids = unselected_course_df['COURSE_ID'].values

    #getting recomendation score for each course
    for i in range(unselected_course_matrix.shape[0]):
      result = np.dot(unselected_course_matrix[i],user_vector)
      res[unselected_course_ids[i]] =  result

    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def KMeans_recommendations(clusters_users, user_id):
    tuser_df = clusters_users[clusters_users['user'] == user_id]
    tuser_courses = tuser_df['item'].to_list()
    tuser_cluster = tuser_df['cluster'].iloc[0]
    all_cluster_courses = clusters_users[clusters_users['cluster']==tuser_cluster]['item'].to_list()
    unseen_courses = set(all_cluster_courses).difference(set(tuser_courses))

    courses_cluster = clusters_users[['item', 'cluster']]
    courses_cluster['count'] = [1] * len(courses_cluster)
    courses_cluster = courses_cluster.groupby(['cluster','item']).agg(enrollments = ('count','sum')).reset_index()
    courses_cluster = courses_cluster.sort_values(['cluster','enrollments'], ascending=False)



    tuser_courses_cluster = courses_cluster[courses_cluster['cluster'] == tuser_cluster]
    enrollment_mean_cluster = tuser_courses_cluster['enrollments'].mean()

    # Create a dictionary to store your recommendation results
    res = {}

    for course in unseen_courses:
        course_enrollment = courses_cluster[(courses_cluster['item'] == course) & (courses_cluster['cluster'] == tuser_cluster)]['enrollments'].iloc[0]
        if course_enrollment > enrollment_mean_cluster:
            res[course] =  course_enrollment

            res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


# Model training
def train(model_name, params):
    if model_name == models[2]:
        user_profile_df = load_user_profile()
        features = user_profile_df.loc[:, user_profile_df.columns != 'user']
        km = KMeans(n_clusters = 20, random_state = 123)
        km.fit(features)
        cluster_labels = km.labels_
        user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
        clusters = combine_cluster_labels(cluster_labels, user_ids)
        return clusters

    if model_name == models[3]:
        user_profile_df = load_user_profile()
        user_ids = user_profile_df.loc[:, user_profile_df.columns == 'user']
        features = user_profile_df.loc[:, user_profile_df.columns != 'user']
        #Selecting 4 components to expalin 90% of the variance
        pca_n4 = PCA(n_components = 4)
        features_pca = pca_n4.fit_transform(features)
        features_pca_df = pd.DataFrame(features_pca)

        km = KMeans(n_clusters = 20, random_state = 123)
        km.fit(features_pca)
        cluster_labels = km.labels_
        clusters = combine_cluster_labels(cluster_labels, user_ids)
        return clusters
    
    if model_name == models[4]:
        # user_id, idx_id_dict, id_idx_dict = passing()
        ratings_df = load_ratings()
        num_users = len(ratings_df['user'].unique())
        num_items = len(ratings_df['item'].unique())
        embedding_size = 16
        encoded_data, user_idx2id_dict, course_idx2id_dict = process_dataset_nn(ratings_df)
        x_train, x_val, y_train, y_val = generate_train_dataset_nn(encoded_data)  
        optimizer = keras.optimizers.Adam()
        losses = keras.losses.MeanSquaredError()
        metric = keras.metrics.RootMeanSquaredError()
        recommender = RecommenderNet(num_users, num_items, embedding_size)
        recommender.compile(optimizer = optimizer, loss = losses, metrics=metric)
        history = recommender.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), batch_size = 64, epochs = 10, verbose = 1)
        return recommender, user_idx2id_dict, course_idx2id_dict
    
    pass

#Neural Network model
class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        """
           Constructor
           :param int num_users: number of users
           :param int num_items: number of items
           :param int embedding_size: the size of embedding vector
        """
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size

        # Define a user_embedding vector
        # Input dimension is the num_users
        # Output dimension is the embedding size
        self.user_embedding_layer = layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define a user bias layer
        self.user_bias = layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")

        # Define an item_embedding vector
        # Input dimension is the num_items
        # Output dimension is the embedding size
        self.item_embedding_layer = layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        # Define an item bias layer
        self.item_bias = layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")

    def call(self, inputs):
        """
           method to be called during model fitting

           :param inputs: user and item one-hot vectors
        """
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability
        return tf.nn.relu(x)

def process_dataset_nn(raw_data, data_dict):

    encoded_data = raw_data.copy()
    encoded_data = encoded_data[['user','item', 'rating']]
    data_dict = data_dict

    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    # Mapping course ids to indices
    course_list = data_dict['COURSE_ID'].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")

    return encoded_data, user_idx2id_dict, course_idx2id_dict, user_id2idx_dict, course_id2idx_dict

def generate_train_dataset_nn(dataset, scale=True):

    min_rating = min(dataset["rating"])
    max_rating = max(dataset["rating"])

    dataset = dataset.sample(frac=1, random_state=42)
    x = dataset[["user", "item"]].values
    if scale:
        y = dataset["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    else:
        y = dataset["rating"].values

    # Assuming training on 80% of the data and validating on 10%, and testing 10%
    train_indices = int(0.8 * dataset.shape[0])

    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    return x_train, x_val, y_train, y_val


def nn_test_dataset(idx_id_dict, id_idx_dict, enrolled_course_ids, course_genres_df, user_id,user_id2idx_dict,course_id2idx_dict):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # First find all enrolled courses for user
    unselected_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unselected_course_ids)]
    unseen = unselected_course_df['COURSE_ID'].values
    
    temp_dict = {}
    users = []
    items = []
    users = [user_id]*len(unselected_course_ids)
    temp_dict['user'] = users
    

    for course in unseen:
        items.append(course)

    temp_dict['item'] = items
    temp_dict_df = pd.DataFrame(temp_dict)
    temp_dict_df['user'] = temp_dict_df['user'].map(user_id2idx_dict)
    temp_dict_df['item'] = temp_dict_df['item'].map(course_id2idx_dict)
    array = temp_dict_df[["user", "item"]].values
    array = np.asarray(array).astype('float32')
    return array

       
def idx2id(data,rating_df):
    user_list = rating_df["user"].unique().tolist()
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}

    # Mapping course ids to indices
    course_list = rating_df["item"].unique().tolist()
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}
    # Convert original user ids to idx
    data["user"] = data["user"].map(user_idx2id_dict)
    # Convert original course ids to idx
    data["item"] = data["item"].map(course_idx2id_dict)
    return data


# Prediction
def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()

    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        # Course Similarity model
        if model_name == models[0]:
            sim_matrix = load_course_sims().to_numpy()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        if model_name == models[1]:
            user_profile_df = load_user_profile()
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df['user'] == user_id]
            enrolled_course_ids = user_ratings['item'].to_list()
            user_profile = user_profile_df[user_profile_df['user'] == user_id]
            user_vector = user_profile.iloc[0,1:].values
            course_genre_df = load_course_genre()
            enrolled_course_ids = user_ratings['item'].to_list()
            res = user_profile_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, course_genre_df,user_vector)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
        
        if (model_name == models[2] or model_name == models[3]):
            clusters = train(model_name, params)
            user_profile_df = load_user_profile()
            ratings_df = load_ratings()
            ratings_df = ratings_df[['user', 'item']]
            clusters_users = pd.merge(ratings_df, clusters, left_on='user', right_on='user')
            res = KMeans_recommendations(clusters_users,user_id)
            for key, score in res.items():
                users.append(user_id)
                courses.append(key)
                scores.append(score)
             
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE']).reset_index()
    res_df = res_df.drop_duplicates(subset=['SCORE'], keep = 'last')
    res_df = res_df.rename(columns = {'index':'level_1'})
    top = res_df.groupby('USER')['SCORE'].nlargest(int(params['top_courses'])).reset_index()
    top_res_df = pd.merge(top, res_df, how = 'left', on = ['USER', 'level_1'])
    top_res_df = top_res_df[['USER', 'COURSE_ID', 'SCORE_x']].rename(columns={'SCORE_x':'SCORE'})

    return top_res_df

def predict2(model_name, user_ids, params):
    ratings_df = load_ratings()
    course_genres_df = load_course_genre()
    num_users = len(ratings_df['user'].unique())
    num_items = len(course_genres_df['COURSE_ID'].unique())
    embedding_size = 16
    encoded_data, user_idx2id_dict, course_idx2id_dict,user_id2idx_dict, course_id2idx_dict = process_dataset_nn(ratings_df,course_genres_df)
    x_train, x_val, y_train, y_val = generate_train_dataset_nn(encoded_data)  
    optimizer = keras.optimizers.Adam()
    losses = keras.losses.MeanSquaredError()
    metric = keras.metrics.RootMeanSquaredError()
    recommender = RecommenderNet(num_users, num_items, embedding_size)
    recommender.compile(optimizer = optimizer, loss = losses, metrics=metric)
    recommender.fit(x = x_train, y = y_train, validation_data = (x_val, y_val), batch_size = 64, epochs = 10, verbose = 1)
    for user_id in user_ids:
        idx_id_dict, id_idx_dict = get_doc_dicts()
        users = []
        courses = []
        scores = []
        res_dict = {}
        user_ratings = ratings_df[ratings_df['user'] == user_id]
        enrolled_course_ids = user_ratings['item'].to_list()
        x_test = nn_test_dataset(idx_id_dict, id_idx_dict, enrolled_course_ids, course_genres_df, user_id, user_id2idx_dict,course_id2idx_dict)
        print(x_test)
        pred = recommender.predict(x_test)
        print(pred)

    x_test_id = pd.DataFrame(x_test, columns = ['user', 'item'])
    x_test_id['user'] = x_test_id['user'].map(user_idx2id_dict)
    x_test_id['item'] = x_test_id['item'].map(course_idx2id_dict)
    x_test_id['score'] = pred
    x_test_id.rename(columns = {'item':'COURSE_ID', 'score':'SCORE'}, inplace = True) 
    x_test_id.sort_values(by = 'SCORE', ascending = False, inplace = True)
    top_res_df = x_test_id[:int(params['top_courses'])]
    print(top_res_df)

    return top_res_df
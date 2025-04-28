import pandas as pd
import numpy as np

# sklearn classifier's
from sklearn.neighbors import KNeighborsClassifier # knn algo
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model # LogisticRegression LinearRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans # K-means Clustering
from sklearn.naive_bayes import GaussianNB # naive bayes
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# sklearn misc.
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler        # StandardScaler = normalize features             ——LabelEncoder =  Label Encoding (convert value into unique int) - good for ft. w/ many unique values (to not create many dummy columns)
from sklearn import metrics                                           # accuracy score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV # cross_val_score = average(repeatedly split dataset into training/testing, .fit(), accuracy_score(.predict())         ——train_test_split example:   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=x)
from sklearn.pipeline import make_pipeline                            #
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA # (principal component analysis (unsupervised dimensionality reduction))
from sklearn.manifold import TSNE # PCA alternative (we use for SMOTE noise detection in tight clusters)

# misc.
from imblearn.over_sampling import SMOTE # imbalance handling ('synthetic minority oversampling technique')

# plotting
import matplotlib.pyplot as plt                                       # matplotlib. (graphs/plots)
import matplotlib.cm as cm                                            # generate colors (ie: for loop of auc/roc curves)
import seaborn as sns                                                 # matplotlib alternative

"""Depends on Tensorflow / Keras Version (Comment/Uncomment the appropriate one)"""
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.callbacks import EarlyStopping

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# defaults
pd.set_option('display.max_columns', None) # don't limit # columns shown

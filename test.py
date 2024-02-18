import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_iris
from scipy.stats import chi2_contingency
import re
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


df = pd.read_pickle(r"C:\Users\tamar\Downloads\XY_train.pkl")




# Load Python Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

class tSNE:
    def __init__(self, data, labels):
        dataset = np.concatenate((data, labels), axis=1)
        df = pd.DataFrame(data=data[0:,0:])
        self.df = StandardScaler().fit_transform(df)
        self.df = pd.DataFrame([self.df], list(df.columns) )

    def display(self):
        tsne = TSNE(random_state=0)
        tsne_results = tsne.fit_transform(self.df)
        tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=wine.target)
        plt.show()

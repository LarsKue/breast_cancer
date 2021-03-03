
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main(argv: list) -> int:

    breast = load_breast_cancer()

    features = breast.data
    labels = breast.target

    # normalize features
    features = StandardScaler().fit_transform(features)

    print(np.mean(features), np.std(features))

    features_labels = np.concatenate((features, labels[:, np.newaxis]), axis=1)
    column_names = np.append(breast.feature_names, "label")

    df = pd.DataFrame(features_labels)
    df.columns = column_names

    print(df.head())

    pca = PCA(n_components=16)
    pcomponents = pca.fit_transform(features, labels)

    print(pcomponents.shape)

    cs = np.cumsum(pca.explained_variance_ratio_)

    # for plotting
    nc = 2

    # find the number of components for 99% confidence
    # confidence = 0.99
    # nc = np.argmax(cs > confidence)
    # print(nc)

    # redo the pca with less components
    pca = PCA(n_components=nc)
    pcomponents = pca.fit_transform(features, labels)

    print(pcomponents.shape)

    if nc == 2:
        plt.figure(figsize=(10, 9))
        plt.scatter(pcomponents[:, 0], pcomponents[:, 1], marker=".", c=labels)
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.savefig("plot.pdf")
        plt.show()


    return 0



if __name__ == "__main__":
    import sys
    main(sys.argv)

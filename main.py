from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def to_colours(x):
    return list(map(lambda l: "C1" if l else "C0", ~np.isclose(x, 0)))


def main(argv: list) -> int:
    breast = load_breast_cancer()

    features = breast.data
    labels = breast.target

    features_labels = np.concatenate((features, labels[:, np.newaxis]), axis=1)
    column_names = np.append(breast.feature_names, "label")

    df = pd.DataFrame(features_labels)
    df.columns = column_names

    print(df.head())

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, random_state=0)

    # normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    print(np.mean(train_features), np.std(train_features))
    print(np.mean(test_features), np.std(test_features))

    pca = PCA()
    train_components = pca.fit_transform(train_features)

    print(train_components.shape)

    cs = np.cumsum(pca.explained_variance_ratio_)

    # find the number of components for 99% confidence
    confidence = 0.99
    nc = np.argmax(cs > confidence)
    print(nc)

    # redo the pca with less components
    pca = PCA(n_components=nc)
    train_components = pca.fit_transform(train_features)
    test_components = pca.transform(test_features)

    print(train_components.shape)

    train_colours = to_colours(train_labels)
    test_colours = to_colours(test_labels)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Comparison of Feature Separation in Most Important vs. Least Important Principal Components")
    axes[0].scatter(train_components[:, 0], train_components[:, 1], marker=".", c=train_colours)
    axes[0].scatter(test_components[:, 0], test_components[:, 1], marker=".", c=test_colours)
    axes[0].set_xlabel("PC 0")
    axes[0].set_ylabel("PC 1")
    axes[0].set_title("Most Important")
    axes[0].legend(["Benign", "Malignant"])
    axes[1].scatter(train_components[:, -2], train_components[:, -1], marker=".", c=train_colours)
    axes[1].scatter(test_components[:, 0], test_components[:, 1], marker=".", c=test_colours)
    axes[1].set_xlabel(f"PC {nc - 2}")
    axes[1].set_ylabel(f"PC {nc - 1}")
    axes[1].set_title("Least Important")
    axes[1].legend(["Benign", "Malignant"])

    plt.savefig("plot.pdf")

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(train_components, train_labels)

    # test the classifier
    predictions = clf.predict(test_components)

    incorrect = predictions != test_labels
    error_rate = np.sum(incorrect) / len(predictions)

    # < 10%
    print(f"Error Rate: {100 * error_rate:.2f}%")

    # show the confusion matrix
    conf_mat = pd.crosstab(test_labels, predictions, normalize=True, rownames=["Actual"], colnames=["Predicted"])
    print(conf_mat)

    plt.show()

    return 0


if __name__ == "__main__":
    import sys

    main(sys.argv)

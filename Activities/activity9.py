from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import matplotlib.pylab as plt
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris

def part1():
    # Define the dataset
    data = {
        'Age': [24, 29, 27, 30, 35, 23, 45, 40, 28, 32],
        'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'],
        'Salary': [18080, 46200, 23000, 21100, 21000, 42300, 41220, 20103, 31000, 42000],
        'Deduction': [114, 0, 30, 340, 0, 41, 97, 65, 0, 0],
        'Days-Worked': [31, 30, 30, 29, 31, 29, 30, 30, 31, 29],
        'Performance': ['High', 'High', 'Low', 'High', 'High', 'High', 'Low', 'Low', 'High', 'Low']
    }
    df = pd.DataFrame(data)

    # Convert categorical columns to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['Gender'])
    print(pd)
    # Extract features and labels
    X = df.drop('Performance', axis=1)
    y = df['Performance']

    # Create a decision tree classifier using Gini Index
    clf = DecisionTreeClassifier(criterion='gini').fit(X, y)

    # Display the decision tree rules
    # tree_rules = export_text(clf, feature_names=list(X.columns))
    # print(tree_rules)
    plot_tree(clf, filled=True)
    plt.title("Decision tree trained on all the iris features")
    plt.show()

def part2():
    iris = load_iris()
    plt.figure()
    clf = DecisionTreeClassifier().fit(iris.data, iris.target)
    plot_tree(clf, filled=True)
    plt.title("Decision tree trained on all the iris features")
    plt.show()
    
def part3():
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_graphviz
    import graphviz
    from IPython.display import Image, display

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    # in the following if we not set max_feature, it means there is no limit, so comment of the two lines below
    tree = DecisionTreeClassifier(max_depth=5, random_state=0)
    #tree = DecisionTreeClassifier(random_state=0)

    tree.fit(X_train, y_train)
    print("Accuracy on training set: {:.3f} %".format(tree.score(X_train, y_train)*100))
    print("Accuracy on test set: {:.3f} %".format(tree.score(X_test, y_test)*100))

    # export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
    #                 feature_names=cancer.feature_names, impurity=False, filled=True)
    # with open("tree.dot") as f:
    #     dot_graph = f.read()
    # graphviz.Source(dot_graph).view()

def part4():
    from sklearn.tree import DecisionTreeClassifier, export_graphviz
    import graphviz
    from IPython.display import Image, display
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Define the dataset
    data = {
        'Age': [24, 29, 27, 30, 35, 23, 45, 40, 28, 32],
        'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'],
        'Salary': [18080, 46200, 23000, 21100, 21000, 42300, 41220, 20103, 31000, 42000],
        'Deduction': [114, 0, 30, 340, 0, 41, 97, 65, 0, 0],
        'Days-Worked': [31, 30, 30, 29, 31, 29, 30, 30, 31, 29],
        'Performance': ['High', 'High', 'Low', 'High', 'High', 'High', 'Low', 'Low', 'High', 'Low']
    }

    df = pd.DataFrame(data)

    # Convert categorical columns to numerical using one-hot encoding
    df = pd.get_dummies(df, columns=['Gender'])

    # Extract features and labels
    X = df.drop('Performance', axis=1)
    y = df['Performance']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to train and evaluate the model
    def train_and_evaluate_model(max_depth=None):
        # Create a decision tree classifier using Gini Index
        clf = DecisionTreeClassifier(criterion='gini', max_depth=max_depth)
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy (max_depth={max_depth}): {accuracy}")

        # Display the decision tree rules
        tree_rules = export_graphviz(clf, out_file=None, class_names=['High', 'Low'],
                                    feature_names=list(X.columns), impurity=False, filled=True)

        graph = graphviz.Source(tree_rules)
        graph.render(f"tree_depth_{max_depth}")

        display(Image(filename=f'tree_depth_{max_depth}.png'))

    # Train and evaluate the model with no depth limit
    train_and_evaluate_model()

    # Train and evaluate the model with max_depth=3
    train_and_evaluate_model(max_depth=3)

part4()
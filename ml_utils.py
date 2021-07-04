from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

classifier = None

# define a Gaussain NB classifier
clf = GaussianNB()

# define an SVM classifier
svc = SVC(kernel='poly', degree=3, max_iter=300000)

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)

    # calculate the print the accuracy score
    acc_nb = accuracy_score(y_test, clf.predict(X_test))
    print(f"NB trained with accuracy: {round(acc_nb, 3)}")

    svc.fit(X_train, y_train)
    acc_svc = accuracy_score(y_test, svc.predict(X_test))
    print(f"SVM trained with accuracy: {round(acc_svc, 3)}")

    global classifier
    acc = 0

    if acc_nb >= acc_svc :
        classifier = clf
        acc = acc_nb
    else :
        classifier = svc
        acc = acc_svc

    print("Model chosen: ", classifier, ", with accuracy: ",round(acc, 3))


# function to predict the flower using the model
def predict(query_data):
    print(f"Model used for prediction : {classifier}")
    x = list(query_data.dict().values())
    prediction = classifier.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    classifier.fit(X, y)

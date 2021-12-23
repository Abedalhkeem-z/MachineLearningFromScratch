import numpy as np

class NaiveBayes:
    # Implementation is based on bayes´ theorem.
    def __init__(self):
        self.ALPHA = 1    # The smoothing parameter. prevent the zero problem

def fit(self, feature, targets):
    """
    Fit data and calculate priors
    Args:
        feature (numpy list of float): data variables
        targets (numpt list of int): class label
    """
    self.targets = targets
    number_of_samples, self.number_of_features = feature.shape  # return number of row and column
    self.unique_classes = np.unique(targets)
    number_of_classes = len(self.unique_classes)  # Number of unique classes
    self.priors = np.zeros(number_of_classes, dtype=np.float64)  # Prior for each class

    # Sort features corresponding to its class.
    self.feature_count = np.zeros((len(self.unique_classes), self.number_of_features))
    self.total_count   = np.zeros(number_of_classes)

    for cls in self.unique_classes:
        class_features = feature[cls == targets]
        self.priors[cls] = class_features.shape[0] / float(number_of_samples)  # (Frequency / Total samples) --> P(Class)
        self.feature_count[cls] = np.sum(class_features, axis=0)
        self.total_count[cls] = np.sum(class_features)

def _likelyhood(self, feature_i, feature_index,  cls ):
    """
    P(feature_i|cls) = (Count_of_features_i + alpha)/ ((total_count) + alpha * Total number of features).
     Args:
        feature_i (float): variable value
        feature_index (int): feature index
        cls (int): target position 
    """
    numerator = self.feature_count[cls, feature_index] + self.ALPHA
    denominator = self.total_count[cls] + (self.ALPHA * self.number_of_features)

    return (numerator/denominator) ** feature_i

def likelyhood(self, features, cls):
    """
    P(features|class) = P(feature_1| class) * ........ * P(feature_n| class)
    Args:
        features (numpy list of float): one row of data variables
        cls (int): target 
    """

    lst = []
    for i in range(features.shape[0]):           
        lst.append((self._likelyhood(features[i], i, cls)))

    return np.prod(lst)

def predict(self, feature):
    """
    Args:
        feature (numpy list of float): one row of data variables
    Return:
        predict_prob(a list of float): probability for every class given feature
    """

    predict_prob = np.zeros(len(self.unique_classes))
    joint_likelyhood = np.zeros(len(self.unique_classes))

    for cls in range(len(self.unique_classes)):
        joint_likelyhood[cls] = self.priors[cls] * self.likelyhood(feature, cls)

    # Calculate probability given all classes scores
    denominator = np.sum(joint_likelyhood)

    for cls in range(len(self.unique_classes)):
        numerator = joint_likelyhood[cls]
        predict_prob[cls] = numerator/denominator

    return predict_prob

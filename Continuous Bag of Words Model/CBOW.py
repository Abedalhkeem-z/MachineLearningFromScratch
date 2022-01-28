import numpy as np

"""
Continuous Bag of Words Model
"""

def minibatches(x, y, batch_size):
    # mini batch geneator
    ran_indices = np.random.permutation(x.shape[0])
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        batch_indices = ran_indices[i:i+batch_size]
        yield x[batch_indices], y[batch_indices]

def softmax(X):
    res = np.exp(X - np.max(X, axis=1, keepdims=True))
    return res / res.sum(axis=1, keepdims=True)


class CBOW(object):
    def __init__(self, num_embeddings, output_dim, embedding_dim=125):
        """ 
        Args:
            num_embeddings (int): number of embedding .
            output_dim (int): number of unique class labels
            embedding_dim (int): number of columns of the embedding matrix.
        """

        #Embedding matrix
        self.E = np.random.normal(0, 0.1, size=(num_embeddings, embedding_dim))
        #Weights
        self.W = np.random.uniform(low=-(1/embedding_dim)**1/2, high=(1/embedding_dim)**1/2,
                                        size=(embedding_dim, output_dim))
        #Biases
        self.b = np.random.uniform(low=-(1/embedding_dim)**1/2, high=(1/embedding_dim)**1/2,
                                        size=(1, output_dim))                                

    def forward(self, features):
        """ 
        Args:
            features (numpy list of float): features to be farward from first layer untill softmax
                                            function. 
        Return:
            linear (numpy list of float) : the output from last hidden layer
        """
        # The mean embedding of a spesefic feature
        E_x_mean = np.mean(self.E[features][:], axis=1)
        # linear forward. 
        linear = np.matmul(E_x_mean,self.W) + self.b
        return linear


def train(vocab, train_x, train_y, n_epochs=30, batch_size=24, lr=0.395):
    """ 
    Args:
        vocab (dictionary): ( word (string) : key (id) ) 
        train_x (numpy list of unique words' id): Sentences encoded to ids
        train_y (numpy list of int): the class encoded class label.
        
    Return: 
        model (CBOW)
    """

    #initiate the model
    model = CBOW(len(vocab),5)
    l = np.ones((batch_size))
    # Learn based on number of epochs
    for j in range(n_epochs):
        # Divide x and y to minibatechs
        for X, y in minibatches(train_x,train_y,batch_size):
            # The mean embedding of a spesefic feature
            # Act like forward method
            E_x_mean = np.mean(model.E[X], axis=(1))
            prod = np.matmul(E_x_mean, model.W)
            delta1 = softmax(prod+model.b) - y
            E_x_mean_t = np.transpose(E_x_mean)
            # Update weights biases and the model embedding.
            model.W = model.W - ((lr/batch_size) * (np.matmul(E_x_mean_t, delta1 )))
            model.b = model.b - ((lr/batch_size) * (np.matmul(l,delta1))) 
            delta2 = (1/20) * np.matmul(delta1, np.transpose(model.W))
            for i in range(X.shape[0]):
                model.E[X[i]] = model.E[X[i]] - (lr/batch_size) * delta2[i]

    return model

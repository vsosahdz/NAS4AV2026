"""Extracted from the prior implementation's notebook. Do not edit by hand.

Regenerate with ``python -m nas4av.original.extract --write``; the cell manifest and the
device substitutions live in ``extract.py`` next to this file.

This module is kept as close to its source as running outside Jupyter allows, which is
why it is exempt from linting and type checking. The point is not that the code is good —
several things in it are the subject of the work — but that a reproduction mismatch
should be a finding about the prior results rather than a transcription error of ours.

Two properties are worth knowing before calling anything here.

``Search_Space.metrics`` reports balanced accuracy under the name AUC. Predictions are
binarised at ``logit > 0`` by ``logit2int``, a confusion matrix is built from the 0/1
labels, and ``calc_auc`` averages the two class-wise recalls. No continuous score reaches
the metric.

``train_predict_evaluate`` swallows every exception and records the architecture with
zeros for all measurements. A row of zeros is therefore a failure, not a measurement —
though in the prior campaign's 66,000 rows, none occurred.
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from enum import Enum

import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold


class _NotExtracted:
    """Stands in for classes the manifest deliberately leaves behind.

    ``train_predict`` and ``train_predict_evaluate`` dispatch on the search space with an
    ``isinstance`` chain that includes the graph variants. Leaving those names undefined
    would be worse than it looks: ``train_predict_evaluate`` wraps everything in a bare
    ``except``, so a ``NameError`` would be caught and recorded as an architecture scoring
    zero on every measurement — a failure indistinguishable, in the output table, from a
    real evaluation.

    Binding them here keeps the dispatch well-defined and always false.
    """


Graph_Search_Space = _NotExtracted
Model_FG = _NotExtracted

#: Where tensors and modules live. The prior implementation hardcoded CUDA; this is the
#: only behavioural change the extraction makes.
#:
#: float64 BatchNorm is used throughout, which Apple MPS does not implement, so MPS is
#: deliberately not offered here — it would fail at the first BatchNorm rather than fall
#: back, and a silent CPU fallback would misreport what produced a number.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── debug ─────────────────────────────────────────────────────────────
# code cell 4

def debug(thing, title):
    print('----------------------------------------------------')
    print(title)
    print(thing)
    print('----------------------------------------------------')


# ── split_data ────────────────────────────────────────────────────────
# code cell 5

def split_data(training_data, testing_data):
    """ Transforms the data from DataFrames to lists of documents and labels.
    
    Input:
    training_data: DataFrame -> columns problem id, text1, text2, and label.
    test_data: DataFrame -> same columns.

    Returns:
    X_train: list[str] -> The list of training documents.
    y_train: list[int] -> The list of training labels.
    X_test: list[str] -> The list of test documents.
    y_test = list[int] -> The list of test labels
    """

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(training_data.index)):
        row = training_data.iloc[i]
        X_train += [row['text1']]
        X_train += [row['text2']]
        y_train += [row['label']]

    for i in range(len(testing_data.index)):
        row = testing_data.iloc[i]
        X_test += [row['text1']]
        X_test += [row['text2']]
        y_test += [row['label']]
    return X_train, y_train, X_test, y_test


# ── make_df ───────────────────────────────────────────────────────────
# code cell 9

def make_df(X, Y):
    """ Creates dataframe with list of documents and list of labels.
    
    Input:
    X: list[str] -> a list of documents, where the documents are arranged
                    as known doc1, unknown doc1, known doc2, unknown doc2, etc.
    Y: list[int] -> a list of labels whose length matches the number of problems.
    
    Returns:
    df: DataFrame -> The dataframe with the documents in one column and labels
                     the other column.
    """
    
    X = pd.DataFrame(X, columns=['text'])
    Y = pd.DataFrame(Y, columns=['label'])
    
    df = pd.DataFrame()
    df['text'] = X
    df['label'] = Y

    df = df.fillna(-1)
    df['label'] = df['label'].astype('int')
    
    return df


# ── get_punctuation_bow ───────────────────────────────────────────────
# code cell 19

def get_punctuation_bow(df_training, df_test):
    """Calculate bag of punctuations for the input document DataFrames.
    
    Input:
    df_training: DataFrame -> contains training examples in the single-column
                              format.
    df_test: DataFrame -> contains test examples in the single-column
                              format.

    Returns:
    puncs_train: Tensor -> The bag-of-punctuation marks for the training set.
    puncs_test: Tensor -> The bag-of-punctuation marks for the test set.
    """ 

    training_counts = []
    for i in df_training.index:
        curr_chunk = df_training.loc[i]['text']
        curr_count = [0] * 32
        for char in curr_chunk:
            if char in string.punctuation:
                # The position in the array matches that of punctuation.
                curr_count[string.punctuation.index(char)] += 1
        training_counts += [curr_count]

    test_counts = []
    for i in df_test.index:
        curr_chunk = df_test.loc[i]['text']
        curr_count = [0] * 32
        for char in curr_chunk:
            if char in string.punctuation:
                curr_count[string.punctuation.index(char)] += 1
        test_counts += [curr_count]

    puncs_train = nn.functional.normalize(torch.DoubleTensor(training_counts).to(DEVICE), dim=1)
    puncs_test = nn.functional.normalize(torch.DoubleTensor(test_counts).to(DEVICE), dim=1)

    return puncs_train, puncs_test


# ── RDropout ──────────────────────────────────────────────────────────
# code cell 26

class RDropout(nn.Module):
    """ The implementation of Dropout Layer that ensures reproducibility
    with numpy."""
    
    def __init__(self,
                 percentage,
                 seed=None):
        super().__init__()
        self.percentage = percentage
        self.seed = None

    def forward(self, input, training=True):
        """Only turn off values in the training phase."""
        
        if training == True:
            rows, cols = input.shape

            if self.seed:
                cseed = self.seed
            else:
                cseed = cols
            np.random.seed(cseed)

            mask = torch.tensor(np.random.binomial(1,
                    1 - self.percentage, size=input.shape), device=DEVICE)
            return input * mask
        elif training == False:
            ...
        return input


# ── Model ─────────────────────────────────────────────────────────────
# code cell 28

class Model(nn.Module):
    """Model class leveraging torch nn Module."""
    def __init__(self, 
                 encoding,
                 input_shape,
                 genel,
                 tel,
                 seed=None):
        super().__init__()

        self.seed = seed

        self.device = DEVICE
        self.encoding = encoding
        self.encoding_str = ','.join([str(gene) for gene in encoding])
        self.myparameters = []
        self.tel = tel
        # Gene length.
        self.genel = genel
        self.input_shape = input_shape
        self.flatten_conv = nn.Flatten(1, -1)
        self.identity = nn.Identity()
        self.dropout1 = RDropout(0.9, self.seed)
        self.dropout2 = RDropout(0.6, self.seed)
        self.dropout3 = RDropout(0.3, self.seed)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()
        
        # 7-19 Linears 20-23 convs
        self.int_to_mod = {0: self.identity, 
                           1: self.dropout1, 
                           2: self.dropout2, 
                           3: self.dropout3, 
                           4: self.relu, 
                           5: self.tanh, 
                           6: self.leaky_relu,
                           7: 1,
                           8: 2,
                           9: 4,
                           10: 8,
                           11: 16,
                           12: 32,
                           13: 64,
                           14: 128,
                           15: 256,
                           16: 512,
                           17: 1024,
                           18: 2048,
                           19: 4096,
                           20: (2, 2),
                           21: (2, 8),
                           22: (8, 2),
                           23: (8, 8),}

        self.sequential = self.build_sequential()
        debug(self, 'Model created')

        for simple_param in self.sequential.parameters():
            self.myparameters += [simple_param]

    def build_sequential(self):
        """Use encoding to build the actual PyTorch deep neural network.
        
        Input:
        None

        Returns:
        nn.Sequential -> The sequential model that is represented by this class.
        """
        
        # Modules list.
        mlist = []
        curr_shape = self.input_shape
        for i in range(0, len(self.encoding), self.genel):
            gene = self.encoding[i:i + self.genel][0]
            
            # Get the module corresponding to the integer.
            if gene < 7:
                # In this case (Identity, Dropout, Activation, BatchNorm1D the shape)
                # is unchanged.
                mlist += [self.int_to_mod[gene]]
            elif gene >= 7 and gene <= 19:
                # Linear layers.
                new_shape = self.int_to_mod[gene]
                cl = nn.Linear(curr_shape, new_shape)
                mlist += [cl]
                curr_shape = new_shape
                mlist += [nn.BatchNorm1d(curr_shape, dtype=torch.float64)]

            else:
                # Conv1D layers.
                # I make the rule that the Flatten will follow Conv1D always
                # to ensure that the input channels is always 1.
                kernel_size, channels = self.int_to_mod[gene]
                cc = nn.Conv1d(1, channels, kernel_size)
                mlist += [cc]
                mlist += [self.flatten_conv]
                i += 1
                if kernel_size > curr_shape:
                    new_shape = channels
                else:
                    new_shape = (curr_shape - kernel_size + 1) * channels
                curr_shape = new_shape
                mlist += [nn.BatchNorm1d(curr_shape, dtype=torch.float64)]
            #self.debug(curr_shape, 'curr shape')

        # Binary classification problem, ensure that the output is shape 1.
        mlist += [nn.Linear(curr_shape, 1)]
        return nn.Sequential(*mlist).to(self.device)

    def __str__(self):
        return 'ENCODING +' + str(self.sequential)

    def __eq__(self, other):
        if isinstance(other, Model):
            return self.encoding == other.encoding
        return False
    
    def debug(self, thing, title):
        print("--------------------------------------------------------")
        print(title)
        print(thing)
        print("--------------------------------------------------------")

    def compare_eots_ch(self, X_e, chunks_per_doc, e_length):
        """Combine chunk embeddings that belong to same document and problem.
        
        Input:
        X_e: Tensor -> Embedding tensor for the batch.
        chunks_per_doc: list[int] -> A list showcasing at position i, how 
                                    many chunks were used for document i.
        e_length: int -> The length of the embedding.
        
        Returns:
        X_compared: Tensor -> Contains the embedings of the same problem
                              instance in the same row.

        """

        X_doc_embs = []
        X_compared = []

        X_doc_embs = torch.empty((0, e_length)).to(self.device)

        i = 0
        # Current chunks per document.
        for curr_cpd in chunks_per_doc:
            p1 = torch.empty((0, e_length)).to(self.device)
            for j in range(curr_cpd):
                p1 = torch.cat((p1, X_e[i + j].reshape(1, e_length)), dim=0)
            # The embeddings of the document are combined with the mean.
            p1 = torch.mean(p1, dim=0)
            X_doc_embs = torch.cat((X_doc_embs, 
                                    p1.reshape(1, 
                                               e_length)), 
                                               dim=0)
            i = i + curr_cpd
        del X_e

        X_compared = torch.empty((0, e_length * 2)).to(self.device)

        for k in range(0, X_doc_embs.shape[0] - 1, 2):
            known_emb = X_doc_embs[k]
            unknown_emb = X_doc_embs[k + 1]

            # Concatenate document embeddings of same problem.
            l = torch.cat((known_emb, unknown_emb), dim=0)
            X_compared = torch.cat((X_compared, 
                                    l.reshape(1, 
                                              e_length * 2)), 
                                              dim=0)
        return X_compared


    def forward(self, cpd, embeddings):
        """Forward pass of the deep neural network.
        
        Input:
        cpd: list[int] -> a list that says at position i, how many 
                    chunks were used by document i.
        embeddings: Tensor -> The embeddings of the current batch.

        Returns: 
        out_l: Tensor -> The output of the forward pass.
        """
        
        X_e = embeddings

        # Input compared. 
        # Combine embeddings of same problem instance.
        inp_compd = self.compare_eots_ch(X_e, cpd, self.tel)
        #self.debug(inp_compd, "Problem-level embeddings")       

        # SECTION TO USE SEQUENTIAL
        #out = self.sequential(inp_compd).to(self.device)
        if isinstance(self.sequential[0], nn.Conv1d):
            # Match the shape format for Conv1D with 1 input channel.
            inp_compd = inp_compd.reshape(inp_compd.shape[0], 
                                          1, 
                                          inp_compd.shape[-1])
        
        if isinstance(self.sequential[0], RDropout):
            # The Dropout receives if the Model is in training phase or not.
            out = self.sequential[0](inp_compd, self.training).to(self.device)    
        else:
            out = self.sequential[0](inp_compd).to(self.device)

        '''if not isinstance(self.sequential[0], nn.Identity):
            self.debug(self.sequential[0], 'Current module')
            if hasattr(self.sequential[0], 'weight'):
                    self.debug(self.sequential[0].weight.data, 'Module\'s parameters')
            self.debug(out, 'Current output')'''

        for i, mod in enumerate(self.sequential[1:]):
            if isinstance(mod, nn.Conv1d):
                #self.debug(None, 'in conv reshaping')
                if mod.kernel_size[0] > out.shape[-1]:
                    # In the case where the kernel is larger than the current
                    # output shape, we pad with zeros to keep it valid.
                    out = nn.functional.pad(out, 
                                            (0,
                                             mod.kernel_size[0] - out.shape[-1]))
                out = out.reshape(out.shape[0], 1, out.shape[-1])

            if isinstance(mod, RDropout):
                out = mod(out, self.training).to(self.device) 
            else:
                out = mod(out).to(self.device)

            '''if not isinstance(mod, nn.Identity):
                self.debug(mod, 'Current module')
                if hasattr(mod, 'weight'):
                    self.debug(mod.weight.data, 'Module\'s parameters')
                self.debug(out, 'Current output')
                if isinstance(mod, nn.BatchNorm1d):
                    debug(mod.running_mean, 'run mean')
                    debug(mod.running_var, ' run var')'''

        out_l = out.squeeze(-1).to(self.device)
        #self.debug(out_l.shape, "out shape")

        return out_l   
    
    def parameters(self):
        return self.myparameters

    def weights_init_mod(self, module, seed):
        """Initialize the parameters of the a module, using a numpy seed."""

        module.bias.data = torch.zeros(module.bias.data.shape,
                                       dtype=torch.float64).to(DEVICE)

        np.random.seed(seed)
        module.weight.data = torch.DoubleTensor(\
            self.trunc(np.random.uniform(-1, 
                                        1, 
                                        module.weight.data.shape).\
                                            astype(np.float64), 
                                    decs=4)).to(DEVICE)
        
    def trunc(self, values, decs=0):
        """Function to truncate decimals."""
        return np.trunc(values*10**decs)/(10**decs)

    def weights_init_uniform(self):
        """Initialize own weights with uniform distribution."""
        for ind, mod in enumerate(self.sequential):
            if hasattr(mod, 'weight'):
                if self.seed:
                    cseed = ind * self.seed
                else:
                    cseed = ind
                self.weights_init_mod(mod, cseed)

    @classmethod
    def encoding_str2list(cls, encoding_as_str):
        """Get the integer list encoding from the str version."""
        return [int(gene) for gene in encoding_as_str.split(',')]
        


# ── Search_Space ──────────────────────────────────────────────────────
# code cell 30

class Search_Space():
    """The class with information of the Search Space."""

    def __init__(self):
        self.genel = 1
        self.loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy + logit.
    
    @classmethod
    def logit2int(self, logit):
        """Use decision threshold to get the class prediction."""

        if logit > 0.0:
            return 1.0
        else:
            return 0.0

    def train_model(self,
                    model, 
                    train_embeddings,
                    train_df, 
                    training_cpd,
                    n_epochs, 
                    batch_size, 
                    optimizer="SGD", 
                    lr=0.001,
                    verbose=False,):
        """ Carry out the partial training using the mini-batch approach.
        
        Input:
        model: Model | Model_FG -> A model to train.
        train_embeddings: Tensor -> Embeddings that represent the training set.
        train_df: DataFrame -> DataFrame with documents, labels from the
                               training set.
        training_cpd: list[int] -> List of chunks used per document in the
                                   training set.
        n_epochs: int -> number of epochs.
        batch_size: int -> The batch size.
        optimizer: str -> The optimizer name.
        lr: float -> learning rate.
        verbose: bool -> indicates how much information to display during 
                         training.

        Returns:
        train_losses_per_epoch: list[float] -> Loss values per epoch on the 
                                               training set.
        val_losses_per_epoch: list[float] -> Loss values per epoch on the
                                             validation set (empty in this case).
        """
        
        if verbose:
            self.debug(train_embeddings, 'Training embeddings')
            self.debug(train_df['label'], 'Training y labels')

        train_losses_per_epoch = []
        val_losses_per_epoch = []

        # Instantiate optimizer.
        if optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        elif optimizer == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "SparseAdam":
            optimizer = optim.SparseAdam(model.parameters(), lr=lr)        
        elif optimizer == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=lr)
        elif optimizer == "ASGD":
            optimizer = optim.ASGD(model.parameters(), lr=lr)
        elif optimizer == "LBFGS":
            optimizer = optim.LBFGS(model.parameters(), lr=lr)
        elif optimizer == "NAdam":
            optimizer = optim.NAdam(model.parameters(), lr=lr)
        elif optimizer == "RAdam":
            optimizer = optim.RAdam(model.parameters(), lr=lr)
        elif optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer == "Rprop":
            optimizer = optim.Rprop(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.1)

        self.debug('--------------------------------------------------', '')

        chunk_indices = []
        start_index = 0

        for i in range(0, len(training_cpd), 2):
            num_chunks_pair = training_cpd[i] + training_cpd[i + 1]
            chunk_indices.append((start_index, start_index + num_chunks_pair))
            start_index += num_chunks_pair

        for epoch in range(n_epochs):
            # Times to cover data
            ttcd = len(training_cpd) // 2 // batch_size
            
            if verbose:
                self.debug(epoch, 'epoch')
            else:
                if epoch % 20 == 0:
                    self.debug(epoch, 'epoch')

            epoch_train_losses = []
            for k in range(ttcd):
                # losses epoch
                
                random.seed((k + 1) * (epoch + 1))
                # Selected indices
                s_idxs1 = random.sample(list(range(0, len(training_cpd) // 2)), batch_size)
                if verbose:
                    self.debug(s_idxs1, 'Mini-Batch chosen indices')
                s_idxs1 = sorted(s_idxs1)

                Xbatch = torch.empty((0, model.tel)).to(model.device)
                # Chunks for the current batch
                cfcb = []

                for doc_pair_index in s_idxs1:
                    start, end = chunk_indices[doc_pair_index]
                    num_chunks_doc1 = training_cpd[doc_pair_index * 2]
                    num_chunks_doc2 = training_cpd[doc_pair_index * 2 + 1]
                    cfcb += [num_chunks_doc1, num_chunks_doc2]

                    batch_e = train_embeddings[start:end]
                    Xbatch = torch.cat((Xbatch, 
                                        batch_e.reshape(num_chunks_doc1 + num_chunks_doc2, model.tel).to(model.device)), dim=0)

                y_pred = model(cfcb, embeddings=Xbatch).to(DEVICE)
                if verbose:
                    self.debug(y_pred, 'Batch y predictions logits')
                
                ybatch = train_df.iloc[s_idxs1]["label"].values
                ybatch = torch.DoubleTensor(ybatch).to(DEVICE)
                if verbose:
                    self.debug(ybatch, 'Batch true y labels')
                
                loss = self.loss_fn(y_pred, ybatch).to(DEVICE)
                if verbose:
                    self.debug(loss, f'loss {loss}')
                # retain_graph=True
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
                
                optimizer.step()
                
                epoch_train_losses += [loss.item()]
                optimizer.zero_grad()
            avg_training_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses_per_epoch += [avg_training_loss] 
        return train_losses_per_epoch, val_losses_per_epoch

    def train_model_cv(self,
                   model,
                   train_embeddings,
                   train_df,
                   training_cpd,
                   n_epochs,
                   batch_size,
                   optimizer="SGD",
                   lr=0.001,
                   verbose=False,
                   k_folds=2,
                   patience=35):
        """ Carry out the cross-validation training using the mini-batch approach.
        
        Input:
        model: Model | Model_FG -> A model to train.
        train_embeddings: Tensor -> Embeddings that represent the training set.
        train_df: DataFrame -> DataFrame with documents, labels from the
                               training set.
        training_cpd: list[int] -> List of chunks used per document in the
                                   training set.
        n_epochs: int -> number of epochs.
        batch_size: int -> The batch size.
        optimizer: str -> The optimizer name.
        lr: float -> learning rate.
        verbose: bool -> indicates how much information to display during 
                         training.
        k_folds: int -> Number of folds for k-fold.
        patience: int -> Number of epochs to continue training waiting for 
                         validation loss to improve.

        Returns:
        all_fold_train_losses: list[list[float]] -> Loss values per epoch on the 
                                               training set, for each fold.
        all_fold_val_losses: list[list[float]] -> Loss values per epoch on the
                                             validation set, for each fold.
        """

        # Instantiate optimizer.
        if optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer == "Adadelta":
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
        elif optimizer == "Adagrad":
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
        elif optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr)
        elif optimizer == "SparseAdam":
            optimizer = optim.SparseAdam(model.parameters(), lr=lr)        
        elif optimizer == "Adamax":
            optimizer = optim.Adamax(model.parameters(), lr=lr)
        elif optimizer == "ASGD":
            optimizer = optim.ASGD(model.parameters(), lr=lr)
        elif optimizer == "LBFGS":
            optimizer = optim.LBFGS(model.parameters(), lr=lr)
        elif optimizer == "NAdam":
            optimizer = optim.NAdam(model.parameters(), lr=lr)
        elif optimizer == "RAdam":
            optimizer = optim.RAdam(model.parameters(), lr=lr)
        elif optimizer == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer == "Rprop":
            optimizer = optim.Rprop(model.parameters(), lr=lr)
        elif optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.1)

        self.debug('--------------------------------------------------', 'training with cross-validation')

        chunk_indices = []
        start_index = 0

        for i in range(0, len(training_cpd), 2):
            num_chunks_pair = training_cpd[i] + training_cpd[i + 1]
            chunk_indices.append((start_index, start_index + num_chunks_pair))
            start_index += num_chunks_pair

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)  # Initialize KFold

        all_fold_train_losses = []
        all_fold_val_losses = []

        for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(training_cpd) // 2))):
            if verbose:
                debug(f"Fold {fold+1}/{k_folds}", '') # Track which fold is being trained
                debug(f'{train_indices}', 'train indices')
                debug(f'{val_indices}', 'validation indices')

            fold_train_losses_per_epoch = []
            fold_val_losses_per_epoch = []

            best_val_loss = float('inf')
            epochs_since_improvement = 0

            for epoch in range(n_epochs):
                ttcd = len(train_indices) // batch_size

                if verbose:
                    self.debug(epoch, 'epoch')
                    self.debug((len(train_indices), batch_size, ttcd), 'n training examples, batch size, times to cover')
                else:
                    if epoch % 20 == 0:
                        self.debug(epoch, 'epoch')

                epoch_train_losses = []
                for k in range(ttcd):
                    if verbose:
                        debug(k, 'time to cover training samples')    
                    random.seed((k + 1) * (epoch + 1) + fold * 1000) # Seed with fold number to avoid identical batches across folds
                    s_idxs1 = random.sample(list(train_indices), batch_size) # Sample from train indices
                    s_idxs1 = sorted(s_idxs1)
                    if verbose:
                        debug(s_idxs1, 'selected indices for batch training')

                    Xbatch = torch.empty((0, model.tel)).to(model.device)
                    cfcb = []

                    for doc_pair_index in s_idxs1: 
                        start, end = chunk_indices[doc_pair_index]
                        num_chunks_doc1 = training_cpd[doc_pair_index * 2]
                        num_chunks_doc2 = training_cpd[doc_pair_index * 2 + 1]
                        cfcb += [num_chunks_doc1, num_chunks_doc2]

                        batch_e = train_embeddings[start:end]
                        Xbatch = torch.cat((Xbatch, batch_e.reshape(num_chunks_doc1 + num_chunks_doc2, model.tel).to(model.device)), dim=0)
                    if verbose:
                        debug(cfcb, 'chunks per docs for current batch')
                        debug(Xbatch, 'X batch')
                    y_pred = model(cfcb, embeddings=Xbatch).to(DEVICE)
                    if verbose:
                        debug(y_pred, 'mini batch predictions')
                    ybatch = train_df.iloc[s_idxs1]["label"].values
                    if verbose:
                        debug(ybatch, 'mini batch labels')
                    ybatch = torch.DoubleTensor(ybatch).to(DEVICE)

                    loss = self.loss_fn(y_pred, ybatch).to(DEVICE)
                    if verbose:
                        debug(loss, 'loss value')
                    loss.backward()
                    # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                    # if verbose:
                    #     debug(model.parameters()[-3:], 'parameters')
                    #     debug([param.grad for param in model.parameters()[-3:]], 'gradients')

                    optimizer.step()
                    # if verbose:
                    #     debug(model.parameters()[-3:], 'parameters')
                    epoch_train_losses += [loss.item()]
                    optimizer.zero_grad()

                avg_training_loss = sum(epoch_train_losses) / len(epoch_train_losses)
                if verbose:
                    debug(avg_training_loss, 'average of mini batch losses')
                fold_train_losses_per_epoch += [avg_training_loss]

                Xbatch_val = torch.empty((0, model.tel)).to(model.device)
                cfcb_val = []
                for doc_pair_index in val_indices:
                    start, end = chunk_indices[doc_pair_index]
                    num_chunks_doc1 = training_cpd[doc_pair_index * 2]
                    num_chunks_doc2 = training_cpd[doc_pair_index * 2 + 1]
                    cfcb_val += [num_chunks_doc1, num_chunks_doc2]
                    batch_e = train_embeddings[start:end]
                    Xbatch_val = torch.cat((Xbatch_val, batch_e.reshape(num_chunks_doc1 + num_chunks_doc2, model.tel).to(model.device)), dim=0)

                y_pred_val = model(cfcb_val, embeddings=Xbatch_val).to(model.device)
                ybatch_val = train_df.iloc[val_indices]["label"].values
                ybatch_val = torch.DoubleTensor(ybatch_val).to(model.device)

                val_loss = self.loss_fn(y_pred_val, ybatch_val).to(model.device)
                fold_val_losses_per_epoch.append(val_loss.item())

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())
                    if verbose:
                        debug(model.parameters()[-3:], 'parameters of improved ')
                        debug(list(best_model_state.values())[-3:], 'parameters of improved in state dict')
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= patience:
                    if verbose:
                        debug(model.parameters()[-3:], 'patience exhausted now parameters ')
                    model.load_state_dict(best_model_state)
                    if verbose:
                        debug(model.parameters()[-3:], 'reloaded parameters ')
                    break
            
            all_fold_train_losses.append(fold_train_losses_per_epoch)
            all_fold_val_losses.append(fold_val_losses_per_epoch) 
        return all_fold_train_losses, all_fold_val_losses

    def debug(self, something, title):
        print("------------------------------------------------------")
        print(title)
        print(something)
        print("------------------------------------------------------")

    def __str__(self):
        return f'Search space of gene length {self.genel}'
    
    def g(self, x, y):
        """Auxiliary function to calculate AUC."""
        if y > 0:
            return x / y
        else:
            return 1

    def calc_auc(self, conf_mat):
        return (self.g(conf_mat[0, 0], 
                       conf_mat[0, 0] + conf_mat[0, 1]) + \
                       self.g(conf_mat[1, 1],
                              conf_mat[1, 1] + conf_mat[1, 0])) / 2
    
    def metrics(self, y_pred, y_true):
        """Measurements to evaluate the performance of Models.
        
        Input:
        y_pred: Tensor -> Predicted label values.
        y_true: Tensor -> Real label values.

        Returns:
        rep: str -> Classification Report of metrics.
        acc: float -> Accuracy
        (prec[1], rec[1], fscore[1], auc): tuple[float] -> Precision, recall
                                                           fscore and AUC.
        """

        y_pred, y_true = np.array(y_pred,
                                  dtype=np.int8), np.array(y_true,
                                                           dtype=np.int8)
        rep = classification_report(y_true, y_pred)
        prec, rec, fscore, _ = precision_recall_fscore_support(y_true, y_pred)
        acc = np.sum(y_pred == y_true) / (len(y_true))
        conf_mat = confusion_matrix(y_true, y_pred)
        auc = self.calc_auc(conf_mat)
        return rep, acc, (prec[1], rec[1], fscore[1], auc)
       



    


# ── Sequential_Search_Space ───────────────────────────────────────────
# code cell 32

class Sequential_Search_Space(Search_Space):
    """ Special case of Search Space for Sequential DNNs size 15."""

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.encoding_size = 15
        self.model = Model([0] * (self.encoding_size), 
                           self.input_shape,
                           self.genel,
                           self.input_shape // 2)

    def sequential_predict(self, m, X, cpd):
        """ Apply forward pass and obtain class predictions."""
        preds = m(cpd, embeddings=X).to(DEVICE)
        preds = [self.logit2int(logit) for logit in preds]
        return preds
    
    def feasible(self, encoding):
        """ Function to determine if a given encoding is feasible.
        
        Input:
        encoding: list[int] -> The encoding of the DNN.

        Returns:
        bool -> True if the encoding is feasible, False otherwise.
        """

        count_21 = encoding.count(21)
        count_23 = encoding.count(23)

        if count_21 + count_23 > 3:
            return False
        
        for i in range(len(encoding)):
            if encoding[i] == 20 or encoding[i] == 21 or encoding[i] == 22 or encoding[i] == 23:
                if i == 0:
                    if encoding[1] > 17:
                        return False
                    
                    k = i + 1
                    while k < len(encoding) and encoding[k] < 7:
                        k += 1
                    if k < len(encoding) and encoding[k] > 17:
                        return False
                    
                elif i == len(encoding) - 1:
                    if encoding[-2] > 17:
                        return False
                    
                    j = i - 1
                    while j >= 0 and encoding[j] < 7:
                        j -= 1
                    if j >= 0 and encoding[j] > 17:
                        return False
                else:
                    if encoding[i - 1] > 17 or encoding[i + 1] > 17:
                        return False
                    
                    k = i + 1
                    while k < len(encoding) and encoding[k] < 7:
                        k += 1
                    if k < len(encoding) and encoding[k] > 17:
                        return False
                    j = i - 1
                    while j >= 0 and encoding[j] < 7:
                        j -= 1
                    if j >= 0 and encoding[j] > 17:
                        return False

        return True


# ── TrainingType ──────────────────────────────────────────────────────
# code cell 44

class TrainingType(Enum):
    """ Enum to indicate the training type."""
    FULL = 'Full'
    PARTIAL = 'Partial'


# ── train_predict ─────────────────────────────────────────────────────
# code cell 45

def train_predict(model,
                  features_train, 
                  features_test,
                  train_df,
                  training_cpd,
                  test_cpd,
                  epochs,
                  batch_size,
                  lr,
                  opt,
                  search_space,
                  tt):
    """ Function that combines training and prediction on both sets.
    
    Input:
    ...
    tt: TrainingType -> The type of training to carry out. 

    Returns:
    y_preds_train: list[float] -> The predictions on the training set.
    y_preds_test: list[float] -> The predictions on the test set.
    elapsed: float -> The time taken to train in seconds.
    """
    
    model.weights_init_uniform()
    start = time.time()

    if tt == TrainingType.FULL:
        _, _ = search_space.train_model_cv(model,
                                        train_embeddings=features_train, 
                                        train_df=train_df, 
                                        training_cpd=training_cpd,
                                        n_epochs=epochs,
                                        batch_size=batch_size,
                                        optimizer=opt,
                                        lr=lr,
                                        verbose=False)
    elif tt == TrainingType.PARTIAL:
        _, _ = search_space.train_model(model,
                                        train_embeddings=features_train, 
                                        train_df=train_df, 
                                        training_cpd=training_cpd,
                                        n_epochs=epochs,
                                        batch_size=batch_size,
                                        optimizer=opt,
                                        lr=lr,
                                        verbose=False)
    model.eval()
    elapsed = time.time() - start
    with torch.no_grad():
        if isinstance(search_space, Sequential_Search_Space):
            y_preds_train = search_space.sequential_predict(model,
                                        features_train,
                                        training_cpd,)
            y_preds_test = search_space.sequential_predict(model,
                                        features_test,
                                        test_cpd,)
        elif isinstance(search_space, Graph_Search_Space):
            y_preds_train = search_space.graph_predict(model,
                                        features_train,
                                        training_cpd,)
            y_preds_test = search_space.graph_predict(model,
                                        features_test,
                                        test_cpd,)

    return y_preds_train, y_preds_test, elapsed


# ── train_predict_evaluate ────────────────────────────────────────────
# code cell 46

def train_predict_evaluate(curr_enc, search_space, udd, tt=TrainingType.PARTIAL):
    """ Function to train, predict and evaluate all at once.
    
    Input:
    curr_enc: list[int] -> The encoding of the DNN.
    search_space: Search_Space -> The search space instance.
    udd: dict -> A useful data dictionary, with training hyperparameters.
    tt: TrainingType -> The type of training.csv

    Returns:
    new_row: DataFrame -> A DataFrame row containing the encoding and its 
                          corresponding performance measurements.

    """

    curr_es = ','.join([str(gene) for gene in curr_enc])
    try:
        # Not meant for ensemble.
        #debug('sequential search space', None)
        if isinstance(search_space, Sequential_Search_Space):
            model = Model(curr_enc, 
                        search_space.input_shape, 
                        search_space.genel,
                        search_space.input_shape // 2)
        elif isinstance(search_space, Graph_Search_Space):
            model = Model_FG(curr_enc, 
                        search_space.input_shape, 
                        search_space.genel,
                        search_space.input_shape // 2)            
        #debug(model, 'model created')
        y_preds_train, y_preds_test, elapsed = train_predict(model, 
                                        udd['features_train'],
                                        udd['features_test'],
                                        udd['train_df'],
                                        udd['training_cpd'],
                                        udd['test_cpd'],
                                        udd['epochs'],
                                        udd['batch_size'],
                                        udd['lr'],
                                        udd['opt'],
                                        search_space,
                                        tt)

        _, curr_acc_train, (_, _, _, curr_auctr) = search_space.metrics(y_preds_train, udd['y_train'])

        _, curr_acc_test, (curr_precte, curr_recte, curr_fscorete, curr_aucte) = search_space.metrics(y_preds_test, udd['y_test'])
        curr_acc = curr_aucte * 0.75 + curr_auctr * 0.25
    except Exception as e:
        print(e)
        curr_acc_train, curr_acc_test = 0, 0
        curr_precte, curr_recte, curr_fscorete, curr_aucte = 0, 0, 0, 0
        curr_acc = 0
        elapsed = 0

    new_row = pd.DataFrame()
    new_row['encoding'] = [curr_es]
    new_row['test accuracy'] = [curr_acc_test]
    new_row['training accuracy'] = [curr_acc_train]
    new_row['combined accuracy'] = [curr_acc]
    new_row['test precision'] = [curr_precte]
    new_row['test recall'] = [curr_recte]
    new_row['test fscore'] = [curr_fscorete]
    new_row['test auc'] = [curr_aucte]
    new_row['training time'] = [elapsed]

    return new_row

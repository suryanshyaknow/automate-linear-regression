import numpy as np
import logging as lg
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, ElasticNet, ElasticNetCV

import pickle
import logging as lg


class LinReg:
    """
    A class used to build a Linear Regeression model on the dataset passed as the parameter with its features as well as
    label to be regressed against. And as such, it is presumed that the analysis and the feature engineering has already 
    been done on the dataset.
    This class will standardize the data via StandardScaler() so that all features can be on the same scale and obviously
    so that the model optimization could increase.
    This class also provides the flexibility of building the model with regularization modes viz. `Lasso (L1`, `Ridge (L2)`
    and `ElasticNet` and to compare their accuracies accordingly.
    ...

    Attributes
    ----------
    logging : bool, default=True
        whether to have logs recorded in the console as well as in a separate file or not.
    X_train : numpy.array
        training features after calling the split method.
    X_test : numpy.array
        test features after calling the split method.
    Y_train : numpy.array
        training label after calling the split method.
    Y_test : numpy.array
        test label after calling the split method.

    Methods
    -------
    split(testSize=0.25)
        splits the features and label into based on the size of testSize passed as a parameter.

    build()
        builds the linear regression model.

    buildLasso()
        builds the linear regression model with LASSO (L1) regularization technique.

    buildRidge()
        builds the linear regression model with Ridge (L2) regularization technique.

    buildElasticNet()
        builds the linear regression model with ElasticNet (L3) regularization technique.

    accuracy(mode="Regression")
        returns the accuracy computed by the adjusted R-squared of the model built based on the regularization type.

    predict(test_array, mode="Regression")
        returns the prediction outcome of the record or array passed as the parameter.

    save(mode="Regression)
        saves the model built based on the mode passed as the parameter, locally in the system.
    """

    def __init__(self, dataframe, features, label, logging=True):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            datset on which model is to be built.
        features : pandas.DataFrame
            features of the dataframe passed as the parameter.
        label : pandas.DataFrame
            label of the dataframe passed as the parameter.
        logging : bool
            whether to record logs in a separate file as well as in the console (default is True).
        """
        # Should the user wants to record the logs in the console as well as in the separate .log file
        self.logging = logging
        if logging == True:
            self._logger()
        else:
            pass

        self.df = dataframe
        self.features = features
        self._X = None  # for storing the standardized features
        self._Y = label

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self._scaler = StandardScaler()

        self._lModel = None
        self._l1Model = None
        self._l2Model = None
        self._elasticModel = None

        lg.info("Model building initiates..")

    def _logger(self):
        """
        A method specific to record logs in the console.
        """
        try:
            self.logFile = f"automation_logging.log"

            # removing the log file if already exists so as not to congest it.
            if os.path.exists(self.logFile):
                os.remove(self.logFile)
            lg.basicConfig(filename=self.logFile, level=lg.INFO,
                           format="%(asctime)s %(levelname)s %(message)s")

            # Adding the StreamHandler to record logs in the console.
            self.console_log = lg.StreamHandler()

            # setting level to the console log.
            self.console_log.setLevel(lg.INFO)

            # defining format for the console log.
            self.format = lg.Formatter("%(levelname)s %(asctime)s %(message)s")
            self.console_log.setFormatter(self.format)

            # adding handler to the console log.
            lg.getLogger('').addHandler(self.console_log)

        except Exception as e:
            lg.info(e)

        else:
            lg.info("logger activated!")

    def _standardize(self, data):
        """
        This method standardizes the data without changing its meaning per se, so that all features are on the same scale.

        Parameters
        ----------
        data : dataframe, 2-D array
            Data to be standardized.
        """
        try:
            return self._scaler.fit_transform(data)

        except Exception as e:
            lg.error("LinReg._standardize()", e)
            print(e)

    def _standardize_features(self):
        """
        Returns the features standardized.
        """
        try:
            self._X = self._standardize(self.features)

        except Exception as e:
            lg.error("LinReg._features()", e)
            print(e)

    def split(self, testSize=0.25):
        """
        This method splits the data into Test and Train sub-parts based on the test size passed by the user.

        Parameters
        ----------
        testSize : float, default=0.35
            size of the test size to be casted aside for the testing purposes and for computing the accuracy of the model
            to be built.
        """
        try:
            # First and foremost features are to be standardize()
            self._standardize_features()

            # now, splitting the features and label into train and test sub-data
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self._X, self._Y, test_size=testSize, random_state=100)

        except Exception as e:
            lg.error("LinReg.split()", e)
            print(e)

        else:
            lg.info("data split is done successfully!")

    def build(self):
        """
        A method specific to build the Linear Regression model (without any regularization techniques).
        """
        try:
            self.l_Model = LinearRegression()

            lg.info("readying the model...")
            self._lModel.fit(self.X_train, self.Y_train)
            lg.info("Model executed succesfully!")

        except Exception as e:
            lg.error("LinReg.build()", e)
            print(e)

    def buildLasso(self):
        """
        This method builds the model with Lasso Regression regularization so that error terms are in more control. 
        """
        try:
            # cross validation to compute the best value of alpha
            lassocv = LassoCV(alphas=None, cv=10, max_iter=1000)
            lassocv.fit(self.X_train, self.Y_train)

            lg.info("readying the L1 Model...")
            self._l1Model = Lasso(lassocv.alpha_)
            self._l1Model.fit(self.X_train, self.Y_train)
            lg.info("L1 Model executed!")

        except Exception as e:
            lg.error("LinReg.buildLAsso()", e)
            print(e)

    def buildRidge(self):
        """
        This method builds the model with Ridge Regression Regularization so that error terms are in more control. 
        """
        try:
            # cross validation to compute the best value of alpha
            ridgecv = RidgeCV(cv=10)
            ridgecv.fit(self.X_train, self.Y_train)

            lg.info("readying the L2 Model...")
            self.l2Model = Ridge(ridgecv.alpha_)
            self.l2Model.fit(self.X_train, self.Y_train)
            lg.info("L2 Model executed!")

        except Exception as e:
            lg.error("LinReg.buildRidge()", e)
            print(e)

    def buildElasticNet(self):
        """
        This method builds the model with ElasticNet Regression regularization so that error terms are in more control. 
        """
        try:
            # cross validation to compute the best value of alpha
            en_cv = ElasticNetCV(cv=10)
            en_cv.fit(self.X_train, self.Y_train)

            lg.info("readying the ElasticNet Model...")
            self._elasticModel = ElasticNet(en_cv.alpha_)
            self._elasticModel.fit(self.X_train, self.Y_train)
            lg.info("ElasticNet Model executed!")

        except Exception as e:
            lg.error("LinReg.buildElasticNet()", e)
            print(e)

    def accuracy(self, mode='Regression'):
        """
        This method computes the accuracy of the built model using the `Adjusted R-squared`.

        Parameters
        ----------
        mode : str, default="Regression"
            regularization techniques viz. "L1" for LASSO, "L2" for Ridge, "Elastic" for ElasticNet and default="Regression"
            for no regularization.
        """
        try:
            if mode == 'Elastic':
                r_sq = self._elasticModel.score(self.X_test, self.Y_test) * 100

            elif mode == 'L1':
                r_sq = self._l1Model.score(self.X_test, self.Y_test) * 100

            elif mode == 'L2':
                r_sq = self.l2Model.score(self.X_test, self.Y_test) * 100

            else:
                r_sq = self._lModel.score(self.X_test, self.Y_test) * 100

            n = self.X_test.shape[0]               # number of rows
            p = self.X_test.shape[1]               # number of predictors
            adj_rsquared = 1-(1 - r_sq)*(n-1)/(n-p-1)  # adjusted r-squared

            lg.info(
                f"The {mode} model appears to be {adj_rsquared}% accurate.")
            return round(adj_rsquared, 3)

        except Exception as e:
            lg.error("LinReg.accuracy()", e)
            print(e)

    def predict(self, test_array, mode="Regression"):
        """
        A method specific to yield prediction results based on the mode of the model passed.

        Before pasing the test record(s) into the model for prediction, this method will standardize the test record(s) too, 
        as whatever is made happen to the train and the test data, the same's gotta be done for the test record(s) to yield 
        accurate prediction outcome.

        Parameters
        ----------
        test array : np.array
            test input based on what prediction is to be yielded.
        mode : str, default="Regression"
            decides from what mode of the model, the prediction is to be yielded.
            regularization techniques viz. "L1" for LASSO, "L2" for Ridge, "Elastic" for ElasticNet and default="Regression"
            for no regularization.
        """
        try:
            # conversion of input passed into array (if not already) and reshaping it to be fed into StandardScaler()
            test_array = np.array(test_array).reshape(1, -1)

            # standardize
            std_test_array = self._scaler.transform(test_array)

            if mode == "L1":
                return self._l1Model.predict(std_test_array)
            elif mode == "L2":
                return self.l2Model.predict(std_test_array)
            elif mode == "Elastic":
                return self._elasticModel.predict(std_test_array)
            else:
                return self._lModel.predict(std_test_array)

        except Exception as e:
            lg.error("LinReg.predict()", e)
            print(e)

    def save(self, mode="Regression"):
        """
        The method to save the model locally.

        Parameters
        ----------
        mode : str, default="Regression"
            decides what type of the model is to be saved locally.
        """
        try:
            if mode == "ElasticNet":
                pickle.dump(self._elasticModel, open(f"{self.df}", 'wb'))
            elif mode == "L1":
                pickle.dump(self._l1Model, open(f"{self.df}__lasso.sav", 'wb'))
            elif mode == "L2":
                pickle.dump(self.l2Model, open(f"{self.df}__ridge.sav", 'wb'))
            else:
                pickle.dump(self.l_Model, open(
                    f"{self.df}__elasticnet.sav", 'wb'))

            lg.info(f"The {mode} model is saved at {os.getcwd()} sucessfully!")

        except Exception as e:
            lg.error("LinReg.save()", e)
            print(e)

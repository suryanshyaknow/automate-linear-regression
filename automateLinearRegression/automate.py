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
    This is a class specific to build a Linear Regeression model on the dataset passed as the parameter with its features as well
    as label to be regressed against. And as such it is assumed that the analysis and the feature enigineering is already done on
    on the dataset that is to say all desired conversions are made on the features before passing them as parameters.
    
    This class will also standardizes the data via StandardScaler() so that all features can be on the same scale and goes without 
    saying that the model optimization could increase.

    This class also gives user the flexibility of building the model with regularization modes viz. Lasso (L1), Ridge (L2)
    and ElasticNet.
    """

    def __init__(self, dataframe, features, label, logging=True):
        
        # Should the user wants to record the logs in the console as well as in the separate .log file
        self.logging = logging
        if logging == True:
            self.logger()
        else:
            pass

        self.df = dataframe
        self.features = features
        self.X = None # for storing the standardized features
        self.Y = label

        
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.scaler = StandardScaler()

        self.lModel = None
        self.l1Model = None
        self.l2Model = None
        self.elasticModel = None
        
        lg.info("Model building initiates..")

    def logger(self):
        """
        A method specific to record logs in the console.
        """
        try:
            self.logFile= f"automation_logging.log"
            
            # removing the log file if already exists so as not to congest it.
            if os.path.exists(self.logFile):
                os.remove(self.logFile)
            lg.basicConfig(filename=self.logFile, level=lg.INFO, format="%(asctime)s %(levelname)s %(message)s")
            
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
            lg.info("Log Class successfully executed!")

    def _standardize(self, data):
        """
        This method standardizes the data without changing its meaning per se, so that all features are on the same scale.
        """
        try:
            return self.scaler.fit_transform(data)

        except Exception as e:
            lg.error("LinReg._standardize()", e)
            print(e)

    def _standardize_features(self):
        """
        Returns the features standardized.
        """
        try:
            self.X = self._standardize(self.features)

        except Exception as e:
            lg.error("LinReg._features()", e)
            print(e)

    def split(self, testSize = 0.25):
        """
        This method splits the data into Test and Train sub-parts based on the test size parameterized by the user.
        """
        try:
            # First and foremost features are to be standardize()
            self._standardize_features()

            # now, splitting the features and label into train and test sub-data
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=testSize, random_state=100)
            
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
            self.lModel = LinearRegression()

            lg.info("readying the model...")
            self.lModel.fit(self.X_train, self.Y_train)
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
            self.l1Model = Lasso(lassocv.alpha_)
            self.l1Model.fit(self.X_train, self.Y_train)
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
            self.elasticModel = ElasticNet(en_cv.alpha_)
            self.elasticModel.fit(self.X_train, self.Y_train)
            lg.info("ElasticNet Model executed!")
            
        
        except Exception as e:
            lg.error("LinReg.buildElasticNet()", e)
            print(e)

    def accuracy(self, mode='Regression'):
        """
        This method calculates the accuracy of the built model based on `Adjusted R-squared`.
        """
        try:
            if mode=='Elastic':
                r_sq=self.elasticModel.score(self.X_test, self.Y_test) * 100
            
            elif mode=='L1':
                r_sq=self.l1Model.score(self.X_test, self.Y_test) * 100

            elif mode=='L2':
                r_sq=self.l2Model.score(self.X_test, self.Y_test) * 100
             
            else:
                r_sq=self.lModel.score(self.X_test, self.Y_test) * 100
            
            n=self.X_test.shape[0]               # number of rows
            p=self.X_test.shape[1]               # number of predictors
            adj_rsquared=1-(1 - r_sq)*(n-1)/(n-p-1) # adjusted r-squared

            lg.info(f"The {mode} model appears to be {adj_rsquared}% accurate.")
            return round(adj_rsquared, 3)            

        except Exception as e:
            lg.error("LinReg.accuracy()", e)
            print(e)

    def predict(self, test_array, mode="Regression"):
        """
        A method specific to yield prediction results.

        Before pasing the test record(s) into model to predict the outcome we have to standardize the test record(s) too, as whatever we made 
        happen to the test data, the same's gotta be done for the test record(s) to yield accurate prediction outcome.
        """
        try:
            # conversion of input passed into array (if not already) and reshaping it to be fed into StandardScaler()
            test_array = np.array(test_array).reshape(1,-1)
                      
            # standardize
            std_test_array = self.scaler.transform(test_array)
            
            if mode == "L1":
                return self.l1Model.predict(std_test_array)
            elif mode == "L2":
                return self.l2Model.predict(std_test_array)
            elif mode == "Elastic":
                return self.elasticModel.predict(std_test_array)
            else:
                return self.lModel.predict(std_test_array)

        except Exception as e:
            lg.error("LinReg.predict()", e)
            print(e)

    def save(self, mode="Regression"):
        """
        The method to save the model locally.
        """
        try:
            if mode=="ElasticNet":
                pickle.dump(self.elasticModel, open(f"{self.df}", 'wb'))
            elif mode=="L1":
                pickle.dump(self.l1Model, open(f"{self.df}__lasso.sav", 'wb'))
            elif mode=="L2":
                pickle.dump(self.l2Model, open(f"{self.df}__ridge.sav", 'wb'))
            else:
                pickle.dump(self.lModel, open(f"{self.df}__elasticnet.sav", 'wb'))
                
            lg.info(f"The {mode} model is saved at {os.getcwd()} sucessfully!")

        except Exception as e:
            lg.error("LinReg.save()", e)
            print(e)
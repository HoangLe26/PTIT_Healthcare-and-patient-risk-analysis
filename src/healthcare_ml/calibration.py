import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

class SigmoidCalibrator:
    """Held-out sigmoid calibration; raw model is never refit on calibration rows."""
    def fit(self,p,y):
        self.model_=LogisticRegression(penalty=None,solver='lbfgs',max_iter=1000)
        self.model_.fit(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1),y)
        return self
    def predict(self,p):
        return self.model_.predict_proba(logit(np.clip(p,1e-7,1-1e-7)).reshape(-1,1))[:,1]


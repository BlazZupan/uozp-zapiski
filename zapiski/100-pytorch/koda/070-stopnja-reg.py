import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch.utils.data import TensorDataset, DataLoader

class LassoRegression():
    def __init__(self, lr=0.001, lambda_reg=0.5, epochs=1000, batch_size=32, verbose=0):
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.feature_names_ = None
        self.scaler = StandardScaler()
        self.verbose = verbose
        self.best_lambda = None

    def _soft_threshold(self, param, lmbd):
        with torch.no_grad():
            param.copy_(param.sign() * torch.clamp(param.abs() - lmbd, min=0.0))
    
    def fit_one(self, X, y, lambda_reg=None):
        """Fit the model with a single regularization parameter."""
        if lambda_reg is not None:
            self.lambda_reg = lambda_reg
            
        # Fit scaler only on training data
        X = self.scaler.fit_transform(X)
        
        # pretvori podatke v tenzorje
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # postavi model in optimizator
        self.model = nn.Linear(X.shape[1], 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        # pripravi podatke za paketno učenje
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # učenje modela
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_hat = self.model(X_batch)
                mse_loss = nn.MSELoss()(y_hat, y_batch)
                l1_norm = sum(param.abs().sum() for name, 
                             param in self.model.named_parameters() 
                             if 'weight' in name)
                loss = mse_loss + self.lambda_reg * l1_norm
                loss.backward()
                self.optimizer.step()
                self._soft_threshold(self.model.weight, self.lambda_reg)
                epoch_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch+1:4d}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self

    def score_mse(self, X, y):
        """Calculate MSE score for given data."""
        check_is_fitted(self)
        X = check_array(X)
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        with torch.no_grad():
            y_pred = self.model(X)
            return nn.MSELoss()(y_pred, y).item()

    def fit(self, X, y, lambda_values=[0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]):
        """Fit the model using cross-validation to choose the best regularization parameter."""
        best_lambda = None
        best_mse = float('inf')
        
        # Perform 3-fold cross-validation for each lambda
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for lambda_reg in lambda_values:
            mse_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Fit model with current lambda
                self.fit_one(X_train, y_train, lambda_reg)
                
                # Calculate MSE on validation set
                mse = self.score_mse(X_val, y_val)
                mse_scores.append(mse)
            
            # Calculate average MSE for current lambda
            avg_mse = np.mean(mse_scores)
            if self.verbose:
                print(f"Lambda: {lambda_reg:.4f}, Average MSE: {avg_mse:.4f}")
            
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lambda = lambda_reg
                self.best_lambda = best_lambda
        if self.verbose:
            print(f"\nBest lambda: {best_lambda:.4f} with MSE: {best_mse:.4f}")
        
        # Fit final model with best lambda on entire dataset
        self.fit_one(X, y, best_lambda)
        return self
    
    def predict(self, X):
        # napovedovanje
        check_is_fitted(self)
        X = check_array(X)
        # Transform using the scaler fitted on training data
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        
        with torch.no_grad():
            return self.model(X).numpy().squeeze()  # Ensure 1D output
    
    def get_feature_importance(self):
        # izlušči pomembnosti značilk
        check_is_fitted(self)
        with torch.no_grad():
            weights = self.model.weight.data.squeeze().numpy()
            return weights
    
    def set_feature_names(self, feature_names):
        # nastavi imena značilk
        self.feature_names_ = feature_names
        return self
    
    def print_feature_importance(self):
        # izpiše pomembnosti značilk
        check_is_fitted(self)
        if self.feature_names_ is None:
            raise ValueError("Feature names not set. Call set_feature_names() first.")
            
        with torch.no_grad():
            weights = self.model.weight.data.squeeze().numpy()
            sorted_weights = sorted(zip(weights, self.feature_names_), 
                                 key=lambda x: abs(x[0]), 
                                 reverse=True)
            for weight, name in sorted_weights:
                if abs(weight) > 1e-6:  # Only print non-zero weights
                    print(f"{name:9s}: {weight:7.4f}")

def learn_and_score(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_pred - y_test))
    return mae

# primer uporabe
if __name__ == "__main__":
    # naloži podatke
    df = pd.read_excel('body-fat-brozek.xlsx')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # razdeli podatke na učno in testno množico
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    model = LassoRegression(lr=0.01, epochs=1000,
                            verbose=1)
    model.set_feature_names(df.columns[:-1])
    mae_test = learn_and_score(model, X_train, y_train, X_test, y_test)
    print(f"MAE-test: {mae_test:.2f}, regularizacija: {model.best_lambda:.4f}")
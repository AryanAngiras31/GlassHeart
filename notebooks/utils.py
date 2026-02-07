import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, precision_score, recall_score, f1_score, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from scipy import stats
from scipy.stats import chi2
import matplotlib.pyplot as plt

def feature_selection(X):
    """
    Drop unimportant categorical features as determined by mutual information analysis.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input DataFrame containing features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with unimportant features removed
    """
    # Drop unimportant categorical features as determined by mutual information
    features_to_drop = ['smoking', 'high_blood_pressure', 'platelets', 'diabetes', 'anaemia']
    
    existing_features_to_drop = [col for col in features_to_drop if col in X.columns]
    X = X.drop(columns=existing_features_to_drop, errors='ignore')
    
    return X
    
def feature_engineering(X):
    """
    Create engineered features for heart failure prediction model.
    
    This function adds non-linear terms, interaction terms, and binned categorical
    features to improve model performance.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input DataFrame containing raw features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional engineered features including:
        - age_squared: Age squared for capturing accelerating risk
        - log_cpk: Log-transformed creatinine phosphokinase
        - age_creatinine: Interaction between age and serum creatinine
        - ejection_sodium: Interaction between ejection fraction and serum sodium
        - ckd_stage: Binned chronic kidney disease stages
        - ef_category: Binned ejection fraction categories
    """
    X=X.copy()
    
    # Add non-linear terms
    X['age_squared'] = X['age'] ** 2   # Captures accelerating risk with age
    X['log_cpk'] = np.log1p(X['creatinine_phosphokinase'])   # Handles right-skewed enzyme levels

    # Gives interaction terms
    X['age_creatinine'] = X['age'] * X['serum_creatinine']
    X['ejection_sodium'] = X['ejection_fraction'] * X['serum_sodium']

    # Binning of clinical markers
    X['ckd_stage'] = pd.cut(X['serum_creatinine'],
                    bins=[0, 1.2, 2.0, 3.0, float('inf')],
                    labels=['normal', 'mild', 'moderate', 'severe'])

    X['ef_category'] = pd.cut(X['ejection_fraction'],
                        bins=[0, 35, 50, float('inf')],
                        labels=['severe', 'moderate', 'normal'])
    
    return X

def grid_searchcv(models, param_grids, X_train, y_train):
    """
    Perform grid search cross-validation for multiple models.
    
    This function systematically searches through all combinations of hyperparameters
    for each model to find the optimal configuration.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and their estimator objects
    param_grids : dict
        Dictionary of parameter grids for each model
        
    Returns:
    --------
    dict
        Dictionary of fitted GridSearchCV objects for each model
        
    Note:
    -----
    Requires X_train and y_train to be defined in the global scope
    """
    # Train and tune the models
    grids = {}
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    for model_name, model in models.items():
        grids[model_name] = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        grids[model_name].fit(X_train, y_train)
        best_params = grids[model_name].best_params_
        best_score = np.sqrt(-1 * grids[model_name].best_score_)

        print(f'\nBest parameters for {model_name}: {best_params}')
        print(f'Best RMSE for {model_name}: {best_score}\n')

    return grids

def pr_auc_from_scores(estimator, X, y_true):
    if hasattr(estimator, "predict_proba"):
        y_score = estimator.predict_proba(X)[:, 1]
    elif hasattr(estimator, "decision_function"):
        y_score = estimator.decision_function(X)
    else:
        # fallback (degrades AP to hard labels; should rarely happen)
        y_score = estimator.predict(X)
    return average_precision_score(y_true, y_score)

scoring = {
  'roc_auc': 'roc_auc',
  'pr_auc': pr_auc_from_scores
}

def randomized_searchcv(models, param_distributions, X_train, y_train, n_iter=200, random_state=42):
    """
    Perform random search cross-validation for multiple models.
    
    This function randomly samples hyperparameter combinations from the specified
    distributions, which is often more efficient than grid search for high-dimensional
    parameter spaces.
    
    Parameters:
    -----------
    models : dict
        Dictionary of model names and their estimator objects
    param_distributions : dict
        Dictionary of parameter distributions for each model
    n_iter : int, default=300
        Number of parameter settings sampled
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict
        Dictionary of fitted RandomizedSearchCV objects for each model
        
    Note:
    -----
    Requires X_train and y_train to be defined in the global scope
    """
    # Train and tune the models
    searches = {}
    cv = KFold(n_splits=4, shuffle=True, random_state=random_state)
    
    for model_name, model in models.items():
        searches[model_name] = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_distributions[model_name], 
            n_iter=n_iter,
            cv=cv, 
            scoring=scoring, 
            refit='pr_auc',
            n_jobs=-1, 
            verbose=0,
            random_state=random_state
        )
        
        searches[model_name].fit(X_train, y_train)

    return searches

def evaluate_split(model, X, y, split_name, feature_set):
    p = model.predict_proba(X)[:,1]

    # Create an array of probabilities for the true class for each sample
    p_true = model.predict_proba(X)[np.arange(len(y)), y]   

    # Sum the log of those probabilities
    log_likelihood = np.sum(np.log(p_true))
    return {
        'feature_set': feature_set,          
        'model': type(model).__name__,  
        'roc_auc': roc_auc_score(y, p),
        'pr_auc': average_precision_score(y, p),
        'brier': brier_score_loss(y, p),
        'log_likelihood': log_likelihood
    }

def plot_calibration_curve(y_true, y_prob, model_name, feature_set, ax=None):
    """Plot calibration curve for a single model"""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10
    )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
            label=f"{model_name} ({feature_set})", linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Calibration Plot: {model_name} on {feature_set}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return ax

def calibrate_models(best_models, X_train, y_train, X_val, y_val, feature_set):
    """Calibrate the best models and compare performance"""
    
    calibrated_results = []
    
    for name, model in best_models.items():
        # Fit original model
        model.fit(X_train[feature_set], y_train)
        y_prob_orig = model.predict_proba(X_val[feature_set])[:, 1]
        
        # Calibrate model
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated_model.fit(X_train[feature_set], y_train)
        y_prob_cal = calibrated_model.predict_proba(X_val[feature_set])[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        
        orig_roc = roc_auc_score(y_val, y_prob_orig)
        cal_roc = roc_auc_score(y_val, y_prob_cal)
        orig_pr = average_precision_score(y_val, y_prob_orig)
        cal_pr = average_precision_score(y_val, y_prob_cal)
        orig_brier = brier_score_loss(y_val, y_prob_orig)
        cal_brier = brier_score_loss(y_val, y_prob_cal)
        
        calibrated_results.append({
            'model': name,
            'original_roc': orig_roc,
            'calibrated_roc': cal_roc,
            'original_pr': orig_pr,
            'calibrated_pr': cal_pr,
            'original_brier': orig_brier,
            'calibrated_brier': cal_brier,
            'roc_improvement': cal_roc - orig_roc,
            'pr_improvement': cal_pr - orig_pr,
            'brier_improvement': orig_brier - cal_brier  # Lower is better
        })
    return pd.DataFrame(calibrated_results)

def likelihood_ratio_test(ll_null, ll_alternative, df_diff):
    """
    Perform a likelihood ratio test between two nested models.
    
    Parameters:
    -----------
    ll_null : float
        Log-likelihood of the null model (simpler model)
    ll_alternative : float
        Log-likelihood of the alternative model (more complex model)
    df_diff : int
        Difference in degrees of freedom between the two models
        
    Returns:
    --------
    dict
        Dictionary containing:
        - lr_stat: Likelihood ratio test statistic
        - p_value: p-value of the test
        - significant: Boolean indicating if the result is significant at alpha=0.05
    """
    # Calculate the uncorrected LR statistic
    lr_stat_uncorrected = -2 * (ll_null - ll_alternative)

    if lr_stat_uncorrected < 0:
        lr_stat = 0.0
        p_value = 1.0  # Cannot reject null hypothesis (no significant difference)
    else:
        lr_stat = lr_stat_uncorrected
        # Use a check to prevent potential issue if df_diff is zero or less
        if df_diff <= 0:
            p_value = 1.0
        else:
            p_value = 1 - chi2.cdf(lr_stat, df_diff)
    
    return {
        'lr_stat': lr_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def delongs_test(y_true, y_pred_proba1, y_pred_proba2):
    """
    Perform DeLong's test to compare the AUCs of two models.
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba1 : array-like, shape (n_samples,)
        Predicted probabilities for model 1
    y_pred_proba2 : array-like, shape (n_samples,)
        Predicted probabilities for model 2
        
    Returns:
    --------
    dict
        Dictionary containing:
        - auc1: AUC of model 1
        - auc2: AUC of model 2
        - z_score: Z-score of the test
        - p_value: Two-tailed p-value
        - significant: Boolean indicating if the difference is significant at alpha=0.05
    """
    # Calculate AUCs
    auc1 = roc_auc_score(y_true, y_pred_proba1)
    auc2 = roc_auc_score(y_true, y_pred_proba2)
    
    # Get number of positive and negative samples
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    
    # Calculate components for DeLong's test
    def calculate_components(y_true, y_pred_proba):
        order = np.argsort(y_pred_proba)
        y_pred_sorted = y_pred_proba[order]
        y_true_sorted = y_true[order]
        
        # Calculate sensitivity (TPR) and 1-specificity (FPR)
        tpr = np.cumsum(y_true_sorted[::-1]) / np.sum(y_true_sorted)
        fpr = np.cumsum(1 - y_true_sorted[::-1]) / np.sum(1 - y_true_sorted)
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        # Calculate components for variance
        v10 = np.zeros(len(y_true))
        v01 = np.zeros(len(y_true))
        
        for i in range(len(y_true)):
            if y_true[i] == 1:
                v10[i] = np.mean(y_pred_proba > y_pred_proba[i])  # P(pred_j > pred_i | y_i=1)
            else:
                v01[i] = 1 - np.mean(y_pred_proba <= y_pred_proba[i])  # P(pred_j <= pred_i | y_i=0)
        
        return auc, v10, v01
    
    # Calculate components for both models
    auc1, v10_1, v01_1 = calculate_components(y_true, y_pred_proba1)
    auc2, v10_2, v01_2 = calculate_components(y_true, y_pred_proba2)
    
    # Calculate covariance terms
    S10 = np.cov(v10_1, v10_2, ddof=1)[0, 1]
    S01 = np.cov(v01_1, v01_2, ddof=1)[0, 1]
    S = S10 / n_pos + S01 / n_neg
    
    # Calculate Z-score
    z = (auc1 - auc2) / np.sqrt(S)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'auc1': auc1,
        'auc2': auc2,
        'z_score': z,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def hosmer_lemeshow_test(y_true, y_pred_proba, n_bins=10):
    """
    Perform the Hosmer-Lemeshow goodness-of-fit test 
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba : array-like, shape (n_samples,)
        Predicted probabilities
    n_bins : int, optional (default=10)
        Number of bins to use for the test
        
    Returns:
    --------
    dict
        Dictionary containing:
        - statistic: Hosmer-Lemeshow test statistic
        - p_value: p-value of the test
        - significant: Boolean indicating if the model is poorly calibrated (p < 0.05)
    """
    # Sort by predicted probabilities
    order = np.argsort(y_pred_proba)
    y_pred_sorted = y_pred_proba[order]
    y_true_sorted = y_true[order]
    
    # Create bins with approximately equal number of samples
    bin_size = len(y_true) // n_bins
    hl_stat = 0
    
    for i in range(n_bins):
        # Get the indices for this bin
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        # Get the actual and predicted values for this bin
        y_bin = y_true_sorted[start:end]
        y_pred_bin = y_pred_sorted[start:end]
        
        # Calculate observed and expected values
        observed = np.sum(y_bin)
        expected = np.sum(y_pred_bin)
        
        # Calculate the denominator for the test statistic
        denominator = expected * (1 - expected / len(y_bin))
        
        # Avoid division by zero
        if denominator > 0:
            hl_stat += ((observed - expected) ** 2) / denominator
    
    # Calculate p-value from chi-square distribution
    p_value = 1 - chi2.cdf(hl_stat, n_bins - 2)
    
    return {
        'statistic': hl_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def find_optimal_threshold(y_true, y_pred_proba, model_name, plot=True):
    """
    Find the optimal threshold using Youden's J statistic.
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba : array-like, shape (n_samples,)
        Predicted probabilities
    plot : bool, optional (default=True)
        Whether to plot the ROC curve and optimal threshold
        
    Returns:
    --------
    dict
        Dictionary containing:
        - optimal_threshold: Threshold that maximizes Youden's J
        - youden_j: Maximum Youden's J value
        - sensitivity: Sensitivity at optimal threshold
        - specificity: Specificity at optimal threshold
        - fpr: False positive rates for ROC curve
        - tpr: True positive rates for ROC curve
        - thresholds: Thresholds used for ROC curve
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate Youden's J statistic (sensitivity + specificity - 1)
    youden_j = tpr - fpr
    
    # Find optimal threshold
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate sensitivity and specificity at optimal threshold
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    
    # Plot ROC curve and optimal threshold if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                   label=f'Optimal threshold = {optimal_threshold:.3f}\nSensitivity = {sensitivity:.3f}\nSpecificity = {specificity:.3f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve with Optimal Threshold for {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
    return {
        'model': model_name,
        'optimal_threshold': optimal_threshold,
        'youden_j': youden_j[optimal_idx],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def compare_models_with_tests(y_true, y_pred_proba_base, y_pred_proba_extended, ll_base, ll_extended, df_diff, model_names=('Base Model', 'Extended Model')):
    """
    Compare two models using likelihood ratio test and DeLong's test.
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0 or 1)
    y_pred_proba_base : array-like, shape (n_samples,)
        Predicted probabilities from the base model
    y_pred_proba_extended : array-like, shape (n_samples,)
        Predicted probabilities from the extended model
    ll_base : float
        Log-likelihood of the base model
    ll_extended : float
        Log-likelihood of the extended model
    df_diff : int
        Difference in degrees of freedom between the models
    model_names : tuple, optional (default=('Base Model', 'Extended Model'))
        Names of the models for display purposes
        
    Returns:
    --------
    dict
        Dictionary containing the results of both tests
    """
    # Perform likelihood ratio test
    lr_result = likelihood_ratio_test(ll_base, ll_extended, df_diff)
    
    # Perform DeLong's test
    delong_result = delongs_test(y_true, y_pred_proba_base, y_pred_proba_extended)
    
    return {
        'likelihood_ratio_test': lr_result,
        'delongs_test': delong_result
    }

def final_evaluation(y_test, y_prob_calibrated, y_pred_hard):
    return {
        # Continuous Metrics (using probabilities)
        'roc_auc': roc_auc_score(y_test, y_prob_calibrated),
        'pr_auc': average_precision_score(y_test, y_prob_calibrated),
        'brier_score': brier_score_loss(y_test, y_prob_calibrated),
        
        # Classification Metrics (using hard predictions)
        'accuracy': accuracy_score(y_test, y_pred_hard),
        'recall': recall_score(y_test, y_pred_hard),
        'precision': precision_score(y_test, y_pred_hard),
        'specificity': recall_score(y_test, y_pred_hard, pos_label=0),
        'f1': f1_score(y_test, y_pred_hard)
    }
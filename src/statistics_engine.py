"""
SLOOZE ADVANCED STATISTICAL ANALYSIS MODULE
============================================
Comprehensive statistical analysis with hypothesis testing,
distribution analysis, and advanced metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    shapiro, normaltest, kstest, anderson, jarque_bera,
    pearsonr, spearmanr, kendalltau, pointbiserialr,
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    f_oneway, kruskal, friedmanchisquare,
    chi2_contingency, fisher_exact,
    levene, bartlett, fligner
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import List, Dict, Any, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DescriptiveStatistics:
    """Comprehensive descriptive statistics calculator."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        
    def calculate_all(self) -> Dict[str, Any]:
        """Calculate all descriptive statistics."""
        self._basic_stats()
        self._central_tendency()
        self._dispersion_measures()
        self._shape_measures()
        self._position_measures()
        return self.results
    
    def _basic_stats(self):
        """Basic count statistics."""
        self.results['basic'] = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': int(self.df.isnull().sum().sum()),
            'missing_percentage': round(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100, 2),
            'duplicate_rows': int(self.df.duplicated().sum()),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
    def _central_tendency(self):
        """Measures of central tendency."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        central = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                central[col] = {
                    'mean': round(float(data.mean()), 4),
                    'median': round(float(data.median()), 4),
                    'mode': round(float(data.mode().iloc[0]), 4) if len(data.mode()) > 0 else None,
                    'trimmed_mean_10': round(float(stats.trim_mean(data, 0.1)), 4),
                    'trimmed_mean_20': round(float(stats.trim_mean(data, 0.2)), 4),
                    'geometric_mean': round(float(stats.gmean(data[data > 0])), 4) if (data > 0).all() else None,
                    'harmonic_mean': round(float(stats.hmean(data[data > 0])), 4) if (data > 0).all() else None,
                    'weighted_mean': round(float(np.average(data, weights=np.arange(1, len(data)+1))), 4)
                }
        self.results['central_tendency'] = central
        
    def _dispersion_measures(self):
        """Measures of dispersion/variability."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        dispersion = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                
                dispersion[col] = {
                    'variance': round(float(data.var()), 4),
                    'std_dev': round(float(data.std()), 4),
                    'std_error': round(float(data.sem()), 4),
                    'range': round(float(data.max() - data.min()), 4),
                    'iqr': round(float(iqr), 4),
                    'mad': round(float(data.mad()), 4) if hasattr(data, 'mad') else round(float((data - data.median()).abs().median()), 4),
                    'coefficient_of_variation': round(float(data.std() / data.mean() * 100), 4) if data.mean() != 0 else None,
                    'quartile_deviation': round(float(iqr / 2), 4),
                    'range_to_mean_ratio': round(float((data.max() - data.min()) / data.mean()), 4) if data.mean() != 0 else None
                }
        self.results['dispersion'] = dispersion
        
    def _shape_measures(self):
        """Measures of shape (skewness, kurtosis)."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        shape = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 3:
                skew = stats.skew(data)
                kurt = stats.kurtosis(data)
                
                # Interpret skewness
                if abs(skew) < 0.5:
                    skew_interp = "Approximately Symmetric"
                elif skew > 0:
                    skew_interp = "Right-Skewed (Positive)"
                else:
                    skew_interp = "Left-Skewed (Negative)"
                
                # Interpret kurtosis
                if abs(kurt) < 0.5:
                    kurt_interp = "Mesokurtic (Normal-like)"
                elif kurt > 0:
                    kurt_interp = "Leptokurtic (Heavy-tailed)"
                else:
                    kurt_interp = "Platykurtic (Light-tailed)"
                
                shape[col] = {
                    'skewness': round(float(skew), 4),
                    'skewness_interpretation': skew_interp,
                    'kurtosis': round(float(kurt), 4),
                    'kurtosis_interpretation': kurt_interp,
                    'excess_kurtosis': round(float(kurt), 4),
                    'fisher_skewness': round(float(skew), 4),
                    'bowley_skewness': round(float((data.quantile(0.75) + data.quantile(0.25) - 2*data.median()) / (data.quantile(0.75) - data.quantile(0.25))), 4) if (data.quantile(0.75) - data.quantile(0.25)) != 0 else None
                }
        self.results['shape'] = shape
        
    def _position_measures(self):
        """Measures of position (percentiles, quantiles)."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        position = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                pct_values = {f'p{p}': round(float(data.quantile(p/100)), 4) for p in percentiles}
                
                deciles = {f'd{i}': round(float(data.quantile(i/10)), 4) for i in range(1, 10)}
                
                position[col] = {
                    'min': round(float(data.min()), 4),
                    'max': round(float(data.max()), 4),
                    **pct_values,
                    **deciles
                }
        self.results['position'] = position


class DistributionAnalyzer:
    """Comprehensive distribution analysis and normality testing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.insights = []
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run all distribution analyses."""
        self._normality_tests()
        self._distribution_fitting()
        self._qq_statistics()
        return self.results
    
    def _normality_tests(self):
        """Multiple normality tests."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        normality = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) >= 8:
                tests = {}
                
                # Shapiro-Wilk (best for n < 5000)
                if len(data) <= 5000:
                    stat, p = shapiro(data)
                    tests['shapiro_wilk'] = {'statistic': round(stat, 6), 'p_value': round(p, 6), 'is_normal': p > 0.05}
                
                # D'Agostino-Pearson
                stat, p = normaltest(data)
                tests['dagostino_pearson'] = {'statistic': round(stat, 6), 'p_value': round(p, 6), 'is_normal': p > 0.05}
                
                # Jarque-Bera
                stat, p = jarque_bera(data)
                tests['jarque_bera'] = {'statistic': round(stat, 6), 'p_value': round(p, 6), 'is_normal': p > 0.05}
                
                # Kolmogorov-Smirnov
                stat, p = kstest(data, 'norm', args=(data.mean(), data.std()))
                tests['kolmogorov_smirnov'] = {'statistic': round(stat, 6), 'p_value': round(p, 6), 'is_normal': p > 0.05}
                
                # Anderson-Darling
                result = anderson(data, dist='norm')
                tests['anderson_darling'] = {
                    'statistic': round(result.statistic, 6),
                    'critical_values': {f'{cv}%': round(sl, 4) for cv, sl in zip(result.significance_level, result.critical_values)},
                    'is_normal': result.statistic < result.critical_values[2]
                }
                
                # Overall conclusion
                normal_count = sum(1 for t in tests.values() if t.get('is_normal', False))
                tests['overall_conclusion'] = {
                    'tests_passed': normal_count,
                    'total_tests': len(tests) - 1,
                    'is_likely_normal': normal_count >= len(tests) // 2
                }
                
                normality[col] = tests
                
                if col == 'price_min':
                    if not tests['overall_conclusion']['is_likely_normal']:
                        self.insights.append(f"Price distribution fails normality tests - consider log transformation")
        
        self.results['normality_tests'] = normality
        
    def _distribution_fitting(self):
        """Fit various distributions and compare."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna()
        distributions = {}
        
        # Test various distributions
        dist_names = ['norm', 'lognorm', 'expon', 'gamma', 'beta', 'weibull_min']
        
        for dist_name in dist_names:
            try:
                dist = getattr(stats, dist_name)
                params = dist.fit(data)
                
                # KS test
                ks_stat, ks_p = kstest(data, dist_name, args=params)
                
                # Log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))
                
                # AIC/BIC
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood
                
                distributions[dist_name] = {
                    'parameters': [round(p, 6) for p in params],
                    'ks_statistic': round(ks_stat, 6),
                    'ks_p_value': round(ks_p, 6),
                    'log_likelihood': round(log_likelihood, 4),
                    'aic': round(aic, 4),
                    'bic': round(bic, 4)
                }
            except Exception as e:
                continue
        
        # Find best fit
        if distributions:
            best_dist = min(distributions.items(), key=lambda x: x[1]['aic'])
            self.results['best_fit_distribution'] = best_dist[0]
            self.insights.append(f"Best fitting distribution: {best_dist[0]} (AIC: {best_dist[1]['aic']:.2f})")
        
        self.results['distribution_fitting'] = distributions
        
    def _qq_statistics(self):
        """Calculate Q-Q plot statistics."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        qq_stats = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) >= 10:
                # Generate theoretical quantiles
                theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                sample = np.sort(data)
                
                # Correlation between theoretical and sample quantiles
                correlation, _ = pearsonr(theoretical, sample)
                
                qq_stats[col] = {
                    'qq_correlation': round(correlation, 6),
                    'is_approximately_normal': correlation > 0.95
                }
        
        self.results['qq_statistics'] = qq_stats


class CorrelationAnalyzer:
    """Advanced correlation analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.insights = []
        
    def analyze_all(self) -> Dict[str, Any]:
        """Run all correlation analyses."""
        self._pearson_correlation()
        self._spearman_correlation()
        self._kendall_correlation()
        self._partial_correlations()
        self._find_strong_correlations()
        return self.results
    
    def _pearson_correlation(self):
        """Pearson correlation with p-values."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        n_cols = len(numeric_df.columns)
        
        corr_matrix = np.zeros((n_cols, n_cols))
        p_matrix = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(numeric_df.columns):
            for j, col2 in enumerate(numeric_df.columns):
                if i <= j:
                    valid_mask = numeric_df[[col1, col2]].notna().all(axis=1)
                    if valid_mask.sum() > 2:
                        r, p = pearsonr(numeric_df.loc[valid_mask, col1], numeric_df.loc[valid_mask, col2])
                        corr_matrix[i, j] = corr_matrix[j, i] = r
                        p_matrix[i, j] = p_matrix[j, i] = p
        
        self.results['pearson'] = {
            'correlation_matrix': pd.DataFrame(corr_matrix, index=numeric_df.columns, columns=numeric_df.columns).to_dict(),
            'p_value_matrix': pd.DataFrame(p_matrix, index=numeric_df.columns, columns=numeric_df.columns).to_dict()
        }
        
    def _spearman_correlation(self):
        """Spearman rank correlation."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr(method='spearman')
        self.results['spearman'] = corr_matrix.to_dict()
        
    def _kendall_correlation(self):
        """Kendall tau correlation."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        # Kendall is slow, so sample if needed
        if len(numeric_df) > 1000:
            numeric_df = numeric_df.sample(1000, random_state=42)
        corr_matrix = numeric_df.corr(method='kendall')
        self.results['kendall'] = corr_matrix.to_dict()
        
    def _partial_correlations(self):
        """Calculate partial correlations."""
        numeric_df = self.df.select_dtypes(include=[np.number]).dropna()
        if len(numeric_df.columns) < 3 or len(numeric_df) < 10:
            return
            
        cols = list(numeric_df.columns)[:5]  # Limit for performance
        partial_corrs = {}
        
        for i, col1 in enumerate(cols):
            for j, col2 in enumerate(cols):
                if i < j:
                    control_vars = [c for c in cols if c not in [col1, col2]]
                    if control_vars:
                        try:
                            # Partial correlation controlling for other variables
                            from sklearn.linear_model import LinearRegression
                            
                            X = numeric_df[control_vars].values
                            y1 = numeric_df[col1].values
                            y2 = numeric_df[col2].values
                            
                            reg1 = LinearRegression().fit(X, y1)
                            reg2 = LinearRegression().fit(X, y2)
                            
                            resid1 = y1 - reg1.predict(X)
                            resid2 = y2 - reg2.predict(X)
                            
                            partial_r, _ = pearsonr(resid1, resid2)
                            partial_corrs[f"{col1}_vs_{col2}"] = round(partial_r, 4)
                        except:
                            continue
        
        self.results['partial_correlations'] = partial_corrs
        
    def _find_strong_correlations(self):
        """Identify and report strong correlations."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        strong_positive = []
        strong_negative = []
        
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                r = corr.iloc[i, j]
                if r > 0.7:
                    strong_positive.append({
                        'var1': corr.columns[i],
                        'var2': corr.columns[j],
                        'correlation': round(r, 4)
                    })
                    self.insights.append(f"Strong positive correlation ({r:.2f}) between {corr.columns[i]} and {corr.columns[j]}")
                elif r < -0.7:
                    strong_negative.append({
                        'var1': corr.columns[i],
                        'var2': corr.columns[j],
                        'correlation': round(r, 4)
                    })
                    self.insights.append(f"Strong negative correlation ({r:.2f}) between {corr.columns[i]} and {corr.columns[j]}")
        
        self.results['strong_correlations'] = {
            'positive': strong_positive,
            'negative': strong_negative
        }


class HypothesisTester:
    """Comprehensive hypothesis testing suite."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.insights = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all hypothesis tests."""
        self._anova_tests()
        self._nonparametric_tests()
        self._variance_tests()
        self._chi_square_tests()
        return self.results
    
    def _anova_tests(self):
        """ANOVA and related tests."""
        if 'category' not in self.df.columns or 'price_min' not in self.df.columns:
            return
            
        groups = [g['price_min'].dropna().values for _, g in self.df.groupby('category')]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) < 2:
            return
            
        # One-way ANOVA
        f_stat, p_val = f_oneway(*groups)
        
        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = all_data.mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Omega-squared (less biased)
        k = len(groups)
        n = len(all_data)
        ms_error = (ss_total - ss_between) / (n - k)
        omega_squared = (ss_between - (k - 1) * ms_error) / (ss_total + ms_error) if (ss_total + ms_error) > 0 else 0
        
        self.results['anova'] = {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_val, 6),
            'is_significant': p_val < 0.05,
            'eta_squared': round(eta_squared, 4),
            'omega_squared': round(omega_squared, 4),
            'effect_size_interpretation': 'Large' if eta_squared > 0.14 else 'Medium' if eta_squared > 0.06 else 'Small'
        }
        
        if p_val < 0.05:
            self.insights.append(f"ANOVA: Significant price differences across categories (F={f_stat:.2f}, p<0.05, eta^2={eta_squared:.3f})")
    
    def _nonparametric_tests(self):
        """Non-parametric tests."""
        if 'category' not in self.df.columns or 'price_min' not in self.df.columns:
            return
            
        groups = [g['price_min'].dropna().values for _, g in self.df.groupby('category')]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) < 2:
            return
        
        # Kruskal-Wallis H-test
        h_stat, p_val = kruskal(*groups)
        
        self.results['kruskal_wallis'] = {
            'h_statistic': round(h_stat, 4),
            'p_value': round(p_val, 6),
            'is_significant': p_val < 0.05
        }
        
        # Pairwise Mann-Whitney U tests
        categories = list(self.df['category'].unique())
        pairwise_tests = {}
        
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1_data = self.df[self.df['category'] == categories[i]]['price_min'].dropna()
                cat2_data = self.df[self.df['category'] == categories[j]]['price_min'].dropna()
                
                if len(cat1_data) >= 2 and len(cat2_data) >= 2:
                    u_stat, p_val = mannwhitneyu(cat1_data, cat2_data, alternative='two-sided')
                    pairwise_tests[f"{categories[i]}_vs_{categories[j]}"] = {
                        'u_statistic': round(u_stat, 4),
                        'p_value': round(p_val, 6),
                        'is_significant': p_val < 0.05
                    }
        
        self.results['mann_whitney_pairwise'] = pairwise_tests
        
    def _variance_tests(self):
        """Tests for equality of variances."""
        if 'category' not in self.df.columns or 'price_min' not in self.df.columns:
            return
            
        groups = [g['price_min'].dropna().values for _, g in self.df.groupby('category')]
        groups = [g for g in groups if len(g) >= 2]
        
        if len(groups) < 2:
            return
        
        # Levene's test
        levene_stat, levene_p = levene(*groups)
        
        # Bartlett's test
        bartlett_stat, bartlett_p = bartlett(*groups)
        
        # Fligner-Killeen test
        fligner_stat, fligner_p = fligner(*groups)
        
        self.results['variance_tests'] = {
            'levene': {'statistic': round(levene_stat, 4), 'p_value': round(levene_p, 6), 'equal_variances': levene_p > 0.05},
            'bartlett': {'statistic': round(bartlett_stat, 4), 'p_value': round(bartlett_p, 6), 'equal_variances': bartlett_p > 0.05},
            'fligner_killeen': {'statistic': round(fligner_stat, 4), 'p_value': round(fligner_p, 6), 'equal_variances': fligner_p > 0.05}
        }
        
    def _chi_square_tests(self):
        """Chi-square tests for categorical associations."""
        if 'category' not in self.df.columns or 'supplier_verified' not in self.df.columns:
            return
            
        # Create contingency table
        contingency = pd.crosstab(self.df['category'], self.df['supplier_verified'])
        
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p_val, dof, expected = chi2_contingency(contingency)
            
            # Cramér's V effect size
            n = contingency.sum().sum()
            min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0
            
            self.results['chi_square'] = {
                'chi2_statistic': round(chi2, 4),
                'p_value': round(p_val, 6),
                'degrees_of_freedom': dof,
                'is_significant': p_val < 0.05,
                'cramers_v': round(cramers_v, 4),
                'effect_size_interpretation': 'Large' if cramers_v > 0.5 else 'Medium' if cramers_v > 0.3 else 'Small'
            }


class OutlierDetector:
    """Multi-method outlier detection."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
        self.insights = []
        
    def detect_all(self) -> Dict[str, Any]:
        """Run all outlier detection methods."""
        self._iqr_method()
        self._zscore_method()
        self._modified_zscore()
        self._isolation_forest()
        self._local_outlier_factor()
        self._elliptic_envelope()
        self._compare_methods()
        return self.results
    
    def _iqr_method(self):
        """IQR-based outlier detection."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        
        # Extreme outliers (3 * IQR)
        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        extreme_outliers = (data < extreme_lower) | (data > extreme_upper)
        
        self.results['iqr'] = {
            'q1': round(q1, 2),
            'q3': round(q3, 2),
            'iqr': round(iqr, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'outlier_count': int(outliers.sum()),
            'outlier_percentage': round(outliers.sum() / len(data) * 100, 2),
            'extreme_outlier_count': int(extreme_outliers.sum())
        }
        
    def _zscore_method(self):
        """Z-score based outlier detection."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna()
        z_scores = np.abs(stats.zscore(data))
        
        outliers_2 = z_scores > 2
        outliers_3 = z_scores > 3
        
        self.results['zscore'] = {
            'outliers_2sigma': int(outliers_2.sum()),
            'outliers_3sigma': int(outliers_3.sum()),
            'percentage_2sigma': round(outliers_2.sum() / len(data) * 100, 2),
            'percentage_3sigma': round(outliers_3.sum() / len(data) * 100, 2)
        }
        
    def _modified_zscore(self):
        """Modified Z-score using MAD."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna()
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            return
            
        modified_z = 0.6745 * (data - median) / mad
        outliers = np.abs(modified_z) > 3.5
        
        self.results['modified_zscore'] = {
            'median': round(median, 2),
            'mad': round(mad, 2),
            'outlier_count': int(outliers.sum()),
            'outlier_percentage': round(outliers.sum() / len(data) * 100, 2)
        }
        
    def _isolation_forest(self):
        """Isolation Forest outlier detection."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna().values.reshape(-1, 1)
        
        for contamination in [0.05, 0.1, 0.15]:
            iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            predictions = iso.fit_predict(data)
            outliers = predictions == -1
            
            self.results[f'isolation_forest_{int(contamination*100)}pct'] = {
                'contamination': contamination,
                'outlier_count': int(outliers.sum()),
                'outlier_percentage': round(outliers.sum() / len(data) * 100, 2)
            }
        
        self.insights.append(f"Isolation Forest (10% contamination) detected {int((predictions == -1).sum())} outliers")
        
    def _local_outlier_factor(self):
        """Local Outlier Factor detection."""
        if 'price_min' not in self.df.columns:
            return
            
        data = self.df['price_min'].dropna().values.reshape(-1, 1)
        
        if len(data) < 20:
            return
        
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        predictions = lof.fit_predict(data)
        outliers = predictions == -1
        
        self.results['lof'] = {
            'n_neighbors': 20,
            'contamination': 0.1,
            'outlier_count': int(outliers.sum()),
            'outlier_percentage': round(outliers.sum() / len(data) * 100, 2)
        }
        
    def _elliptic_envelope(self):
        """Elliptic Envelope (Gaussian) outlier detection."""
        if 'price_min' not in self.df.columns or 'price_max' not in self.df.columns:
            return
            
        data = self.df[['price_min', 'price_max']].dropna()
        
        if len(data) < 10:
            return
        
        try:
            ee = EllipticEnvelope(contamination=0.1, random_state=42)
            predictions = ee.fit_predict(data)
            outliers = predictions == -1
            
            self.results['elliptic_envelope'] = {
                'contamination': 0.1,
                'outlier_count': int(outliers.sum()),
                'outlier_percentage': round(outliers.sum() / len(data) * 100, 2)
            }
        except:
            pass
            
    def _compare_methods(self):
        """Compare all outlier detection methods."""
        comparison = {}
        for method, result in self.results.items():
            if 'outlier_count' in result:
                comparison[method] = result['outlier_count']
        
        if comparison:
            self.results['comparison'] = {
                'method_counts': comparison,
                'average_outliers': round(np.mean(list(comparison.values())), 1),
                'max_outliers': max(comparison.values()),
                'min_outliers': min(comparison.values())
            }


class AdvancedStatisticalEngine:
    """Main orchestrator for all statistical analyses."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.insights = []
        self.results = {}
        
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run all statistical analyses."""
        print("    - Computing descriptive statistics...")
        desc = DescriptiveStatistics(self.df)
        self.results['descriptive'] = desc.calculate_all()
        
        print("    - Analyzing distributions...")
        dist = DistributionAnalyzer(self.df)
        self.results['distributions'] = dist.analyze_all()
        self.insights.extend(dist.insights)
        
        print("    - Computing correlations...")
        corr = CorrelationAnalyzer(self.df)
        self.results['correlations'] = corr.analyze_all()
        self.insights.extend(corr.insights)
        
        print("    - Running hypothesis tests...")
        hypo = HypothesisTester(self.df)
        self.results['hypothesis_tests'] = hypo.run_all_tests()
        self.insights.extend(hypo.insights)
        
        print("    - Detecting outliers...")
        outlier = OutlierDetector(self.df)
        self.results['outliers'] = outlier.detect_all()
        self.insights.extend(outlier.insights)
        
        return self.results

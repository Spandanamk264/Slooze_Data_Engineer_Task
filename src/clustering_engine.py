"""
SLOOZE MACHINE LEARNING CLUSTERING MODULE
==========================================
Advanced clustering, segmentation, and dimensionality reduction.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist, cdist
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, OPTICS, 
    AgglomerativeClustering, MeanShift, SpectralClustering,
    Birch, AffinityPropagation
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from typing import List, Dict, Any, Optional, Tuple
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Feature engineering for clustering."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = []
        self.scalers = {}
        # Drop columns that contain list types (unhashable)
        self._clean_dataframe()
        
    def _clean_dataframe(self):
        """Remove columns with unhashable types."""
        cols_to_drop = []
        for col in self.df.columns:
            try:
                # Check if any value is a list or dict
                sample = self.df[col].dropna().head(10)
                if len(sample) > 0:
                    if isinstance(sample.iloc[0], (list, dict)):
                        cols_to_drop.append(col)
            except:
                pass
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        
    def prepare_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for clustering."""
        self._create_price_features()
        self._create_ratio_features()
        self._create_categorical_encodings()
        self._handle_missing()
        return self.df, self.feature_cols
    
    def _create_price_features(self):
        """Create price-based features."""
        if 'price_min' in self.df.columns and 'price_max' in self.df.columns:
            # Price range
            self.df['price_range'] = self.df['price_max'] - self.df['price_min']
            
            # Price midpoint
            self.df['price_mid'] = (self.df['price_min'] + self.df['price_max']) / 2
            
            # Log prices
            self.df['log_price_min'] = np.log1p(self.df['price_min'])
            self.df['log_price_max'] = np.log1p(self.df['price_max'])
            
            # Price percentile within category
            if 'category' in self.df.columns:
                self.df['price_percentile'] = self.df.groupby('category')['price_min'].rank(pct=True)
            
            self.feature_cols.extend(['price_min', 'price_max', 'price_range', 'price_mid', 'log_price_min', 'log_price_max'])
            
    def _create_ratio_features(self):
        """Create ratio-based features."""
        if 'price_min' in self.df.columns and 'price_max' in self.df.columns:
            # Price ratio
            self.df['price_ratio'] = self.df['price_max'] / (self.df['price_min'] + 1)
            
            # Price volatility proxy
            self.df['price_volatility'] = self.df['price_range'] / (self.df['price_mid'] + 1)
            
            self.feature_cols.extend(['price_ratio', 'price_volatility'])
            
    def _create_categorical_encodings(self):
        """Encode categorical variables."""
        if 'category' in self.df.columns:
            # Category frequency encoding
            cat_freq = self.df['category'].value_counts().to_dict()
            self.df['category_frequency'] = self.df['category'].map(cat_freq)
            
            # Category price rank
            cat_price = self.df.groupby('category')['price_min'].mean().rank().to_dict()
            self.df['category_price_rank'] = self.df['category'].map(cat_price)
            
            self.feature_cols.extend(['category_frequency', 'category_price_rank'])
            
        if 'city' in self.df.columns:
            # City frequency encoding
            city_freq = self.df['city'].value_counts().to_dict()
            self.df['city_frequency'] = self.df['city'].map(city_freq)
            
            self.feature_cols.extend(['city_frequency'])
            
        if 'supplier_verified' in self.df.columns:
            self.df['verified_int'] = self.df['supplier_verified'].astype(int)
            self.feature_cols.append('verified_int')
            
    def _handle_missing(self):
        """Handle missing values in features."""
        for col in self.feature_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
    def get_scaled_features(self, scaler_type='standard') -> np.ndarray:
        """Get scaled feature matrix."""
        feature_df = self.df[self.feature_cols].copy()
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
            
        scaled = scaler.fit_transform(feature_df)
        self.scalers[scaler_type] = scaler
        
        return scaled


class KMeansClusterer:
    """Advanced K-Means clustering with optimization."""
    
    def __init__(self, features: np.ndarray):
        self.features = features
        self.results = {}
        self.insights = []
        self.best_k = None
        self.labels = None
        
    def find_optimal_k(self, k_range: range = range(2, 11)) -> int:
        """Find optimal k using multiple methods."""
        inertias = []
        silhouettes = []
        calinski = []
        davies = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(self.features)
            
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(self.features, labels))
            calinski.append(calinski_harabasz_score(self.features, labels))
            davies.append(davies_bouldin_score(self.features, labels))
        
        # Elbow method - find the elbow point
        deltas = np.diff(inertias)
        delta_deltas = np.diff(deltas)
        elbow_k = list(k_range)[np.argmin(delta_deltas) + 2] if len(delta_deltas) > 0 else 3
        
        # Best silhouette
        silhouette_k = list(k_range)[np.argmax(silhouettes)]
        
        # Lowest Davies-Bouldin
        db_k = list(k_range)[np.argmin(davies)]
        
        self.results['optimization'] = {
            'inertias': {k: round(i, 2) for k, i in zip(k_range, inertias)},
            'silhouette_scores': {k: round(s, 4) for k, s in zip(k_range, silhouettes)},
            'calinski_harabasz': {k: round(c, 2) for k, c in zip(k_range, calinski)},
            'davies_bouldin': {k: round(d, 4) for k, d in zip(k_range, davies)},
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'davies_bouldin_k': db_k
        }
        
        # Vote for best k
        votes = [elbow_k, silhouette_k, db_k]
        self.best_k = max(set(votes), key=votes.count)
        
        self.insights.append(f"Optimal clusters: {self.best_k} (Elbow: {elbow_k}, Silhouette: {silhouette_k}, DB: {db_k})")
        
        return self.best_k
    
    def fit(self, n_clusters: int = None) -> np.ndarray:
        """Fit K-Means with specified or optimal k."""
        if n_clusters is None:
            n_clusters = self.best_k or 5
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        self.labels = kmeans.fit_predict(self.features)
        
        # Cluster analysis
        silhouette_avg = silhouette_score(self.features, self.labels)
        silhouette_per_sample = silhouette_samples(self.features, self.labels)
        
        cluster_stats = {}
        for i in range(n_clusters):
            mask = self.labels == i
            cluster_stats[f'cluster_{i}'] = {
                'size': int(mask.sum()),
                'percentage': round(mask.sum() / len(self.labels) * 100, 2),
                'avg_silhouette': round(silhouette_per_sample[mask].mean(), 4),
                'center': kmeans.cluster_centers_[i].tolist()
            }
        
        self.results['kmeans'] = {
            'n_clusters': n_clusters,
            'inertia': round(kmeans.inertia_, 2),
            'silhouette_score': round(silhouette_avg, 4),
            'calinski_harabasz': round(calinski_harabasz_score(self.features, self.labels), 2),
            'davies_bouldin': round(davies_bouldin_score(self.features, self.labels), 4),
            'cluster_stats': cluster_stats
        }
        
        self.insights.append(f"K-Means clustering: {n_clusters} clusters with silhouette score {silhouette_avg:.3f}")
        
        return self.labels


class DBSCANClusterer:
    """DBSCAN density-based clustering."""
    
    def __init__(self, features: np.ndarray):
        self.features = features
        self.results = {}
        self.insights = []
        self.labels = None
        
    def find_optimal_eps(self, k: int = 5) -> float:
        """Find optimal epsilon using k-distance graph."""
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.features)
        distances, _ = nn.kneighbors(self.features)
        
        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, k-1])
        
        # Find knee point
        deltas = np.diff(k_distances)
        knee_idx = np.argmax(deltas)
        optimal_eps = k_distances[knee_idx]
        
        self.results['eps_optimization'] = {
            'k_neighbors': k,
            'optimal_eps': round(optimal_eps, 4),
            'knee_index': int(knee_idx)
        }
        
        return optimal_eps
    
    def fit(self, eps: float = None, min_samples: int = 5) -> np.ndarray:
        """Fit DBSCAN clustering."""
        if eps is None:
            eps = self.find_optimal_eps()
            
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.features)
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()
        
        self.results['dbscan'] = {
            'eps': round(eps, 4),
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_percentage': round(n_noise / len(self.labels) * 100, 2)
        }
        
        if n_clusters > 1:
            mask = self.labels != -1
            if mask.sum() > n_clusters:
                silhouette = silhouette_score(self.features[mask], self.labels[mask])
                self.results['dbscan']['silhouette_score'] = round(silhouette, 4)
        
        self.insights.append(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({n_noise/len(self.labels)*100:.1f}%)")
        
        return self.labels


class HierarchicalClusterer:
    """Hierarchical/Agglomerative clustering."""
    
    def __init__(self, features: np.ndarray):
        self.features = features
        self.results = {}
        self.insights = []
        self.labels = None
        self.linkage_matrix = None
        
    def compute_linkage(self, method: str = 'ward') -> np.ndarray:
        """Compute linkage matrix."""
        # Sample if too large
        if len(self.features) > 2000:
            idx = np.random.choice(len(self.features), 2000, replace=False)
            sample = self.features[idx]
        else:
            sample = self.features
            
        self.linkage_matrix = linkage(sample, method=method)
        
        # Cophenetic correlation
        c, _ = cophenet(self.linkage_matrix, pdist(sample))
        
        self.results['linkage'] = {
            'method': method,
            'cophenetic_correlation': round(c, 4),
            'sample_size': len(sample)
        }
        
        return self.linkage_matrix
    
    def fit(self, n_clusters: int = 5, linkage_method: str = 'ward') -> np.ndarray:
        """Fit hierarchical clustering."""
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        self.labels = agg.fit_predict(self.features)
        
        silhouette = silhouette_score(self.features, self.labels)
        
        self.results['hierarchical'] = {
            'n_clusters': n_clusters,
            'linkage_method': linkage_method,
            'silhouette_score': round(silhouette, 4),
            'calinski_harabasz': round(calinski_harabasz_score(self.features, self.labels), 2)
        }
        
        self.insights.append(f"Hierarchical clustering: {n_clusters} clusters, silhouette {silhouette:.3f}")
        
        return self.labels


class GMMClusterer:
    """Gaussian Mixture Model clustering."""
    
    def __init__(self, features: np.ndarray):
        self.features = features
        self.results = {}
        self.insights = []
        self.labels = None
        self.probabilities = None
        
    def find_optimal_components(self, n_range: range = range(2, 11)) -> int:
        """Find optimal number of components using BIC/AIC."""
        bic_scores = []
        aic_scores = []
        
        for n in n_range:
            gmm = GaussianMixture(n_components=n, random_state=42, max_iter=200)
            gmm.fit(self.features)
            bic_scores.append(gmm.bic(self.features))
            aic_scores.append(gmm.aic(self.features))
        
        optimal_bic = list(n_range)[np.argmin(bic_scores)]
        optimal_aic = list(n_range)[np.argmin(aic_scores)]
        
        self.results['gmm_optimization'] = {
            'bic_scores': {n: round(b, 2) for n, b in zip(n_range, bic_scores)},
            'aic_scores': {n: round(a, 2) for n, a in zip(n_range, aic_scores)},
            'optimal_bic': optimal_bic,
            'optimal_aic': optimal_aic
        }
        
        return optimal_bic
    
    def fit(self, n_components: int = None, covariance_type: str = 'full') -> np.ndarray:
        """Fit GMM clustering."""
        if n_components is None:
            n_components = self.find_optimal_components()
            
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                              random_state=42, max_iter=200)
        self.labels = gmm.fit_predict(self.features)
        self.probabilities = gmm.predict_proba(self.features)
        
        # Maximum probability for each point (confidence)
        max_probs = self.probabilities.max(axis=1)
        
        self.results['gmm'] = {
            'n_components': n_components,
            'covariance_type': covariance_type,
            'bic': round(gmm.bic(self.features), 2),
            'aic': round(gmm.aic(self.features), 2),
            'converged': gmm.converged_,
            'n_iterations': gmm.n_iter_,
            'avg_confidence': round(max_probs.mean(), 4),
            'min_confidence': round(max_probs.min(), 4)
        }
        
        if len(set(self.labels)) > 1:
            silhouette = silhouette_score(self.features, self.labels)
            self.results['gmm']['silhouette_score'] = round(silhouette, 4)
        
        self.insights.append(f"GMM: {n_components} components, avg confidence {max_probs.mean():.3f}")
        
        return self.labels


class DimensionalityReducer:
    """Dimensionality reduction techniques."""
    
    def __init__(self, features: np.ndarray):
        self.features = features
        self.results = {}
        self.insights = []
        
    def pca(self, n_components: int = None) -> Tuple[np.ndarray, Dict]:
        """Principal Component Analysis."""
        if n_components is None:
            n_components = min(self.features.shape[1], 10)
            
        pca = PCA(n_components=n_components, random_state=42)
        transformed = pca.fit_transform(self.features)
        
        # Explained variance
        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        
        # Find components for 95% variance
        n_95 = np.argmax(cumulative >= 0.95) + 1
        
        self.results['pca'] = {
            'n_components': n_components,
            'explained_variance': [round(e, 4) for e in explained],
            'cumulative_variance': [round(c, 4) for c in cumulative],
            'total_variance_explained': round(cumulative[-1], 4),
            'components_for_95_pct': n_95
        }
        
        self.insights.append(f"PCA: {n_95} components explain 95% variance, {n_components} explain {cumulative[-1]*100:.1f}%")
        
        return transformed, self.results['pca']
    
    def tsne(self, n_components: int = 2, perplexity: float = 30) -> np.ndarray:
        """t-SNE dimensionality reduction."""
        # Sample if too large
        if len(self.features) > 2000:
            idx = np.random.choice(len(self.features), 2000, replace=False)
            sample = self.features[idx]
        else:
            sample = self.features
            idx = np.arange(len(self.features))
            
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42,
                    n_iter=1000, learning_rate='auto', init='pca')
        transformed = tsne.fit_transform(sample)
        
        self.results['tsne'] = {
            'n_components': n_components,
            'perplexity': perplexity,
            'kl_divergence': round(tsne.kl_divergence_, 4),
            'n_iter': tsne.n_iter_,
            'sample_size': len(sample)
        }
        
        return transformed
    
    def umap_like(self, n_components: int = 2) -> np.ndarray:
        """UMAP-like reduction using Isomap."""
        # Sample if too large
        if len(self.features) > 2000:
            idx = np.random.choice(len(self.features), 2000, replace=False)
            sample = self.features[idx]
        else:
            sample = self.features
            
        isomap = Isomap(n_components=n_components, n_neighbors=15)
        transformed = isomap.fit_transform(sample)
        
        self.results['isomap'] = {
            'n_components': n_components,
            'n_neighbors': 15,
            'reconstruction_error': round(isomap.reconstruction_error(), 4)
        }
        
        return transformed


class ClusterProfiler:
    """Profile and interpret clusters."""
    
    def __init__(self, df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]):
        self.df = df.copy()
        self.df['cluster'] = labels
        self.feature_cols = feature_cols
        self.profiles = {}
        self.insights = []
        
    def profile_all(self) -> Dict[str, Any]:
        """Generate complete cluster profiles."""
        self._numeric_profiles()
        self._categorical_profiles()
        self._cluster_names()
        return self.profiles
    
    def _numeric_profiles(self):
        """Profile numeric features per cluster."""
        numeric_stats = {}
        
        for cluster in sorted(self.df['cluster'].unique()):
            if cluster == -1:  # Skip noise
                continue
                
            cluster_df = self.df[self.df['cluster'] == cluster]
            stats = {}
            
            for col in ['price_min', 'price_max', 'price_range']:
                if col in self.df.columns:
                    data = cluster_df[col].dropna()
                    if len(data) > 0:
                        stats[col] = {
                            'mean': round(data.mean(), 2),
                            'median': round(data.median(), 2),
                            'std': round(data.std(), 2),
                            'min': round(data.min(), 2),
                            'max': round(data.max(), 2)
                        }
            
            numeric_stats[f'cluster_{cluster}'] = {
                'size': len(cluster_df),
                'percentage': round(len(cluster_df) / len(self.df) * 100, 2),
                'statistics': stats
            }
        
        self.profiles['numeric'] = numeric_stats
        
    def _categorical_profiles(self):
        """Profile categorical features per cluster."""
        categorical_stats = {}
        
        for cluster in sorted(self.df['cluster'].unique()):
            if cluster == -1:
                continue
                
            cluster_df = self.df[self.df['cluster'] == cluster]
            stats = {}
            
            # Top categories
            if 'category' in self.df.columns:
                top_cats = cluster_df['category'].value_counts().head(3)
                stats['top_categories'] = {cat: int(count) for cat, count in top_cats.items()}
                
            # Top cities
            if 'city' in self.df.columns:
                top_cities = cluster_df['city'].value_counts().head(3)
                stats['top_cities'] = {city: int(count) for city, count in top_cities.items()}
                
            # Verification rate
            if 'supplier_verified' in self.df.columns:
                stats['verified_percentage'] = round(cluster_df['supplier_verified'].mean() * 100, 2)
            
            categorical_stats[f'cluster_{cluster}'] = stats
        
        self.profiles['categorical'] = categorical_stats
        
    def _cluster_names(self):
        """Assign meaningful names to clusters based on characteristics."""
        names = {}
        
        for cluster in sorted(self.df['cluster'].unique()):
            if cluster == -1:
                names[-1] = 'Noise/Outliers'
                continue
                
            cluster_df = self.df[self.df['cluster'] == cluster]
            
            # Determine name based on price
            if 'price_min' in self.df.columns:
                avg_price = cluster_df['price_min'].mean()
                overall_avg = self.df['price_min'].mean()
                
                if avg_price < overall_avg * 0.5:
                    price_tier = 'Budget'
                elif avg_price < overall_avg * 0.8:
                    price_tier = 'Economy'
                elif avg_price < overall_avg * 1.2:
                    price_tier = 'Mid-Range'
                elif avg_price < overall_avg * 1.8:
                    price_tier = 'Premium'
                else:
                    price_tier = 'Enterprise'
                    
                names[cluster] = price_tier
        
        self.profiles['cluster_names'] = names
        self.df['segment_name'] = self.df['cluster'].map(names)
        
        # Generate insights
        for cluster, name in names.items():
            if cluster != -1:
                size = (self.df['cluster'] == cluster).sum()
                pct = size / len(self.df) * 100
                self.insights.append(f"Segment '{name}': {size} products ({pct:.1f}%)")


class AdvancedClusteringEngine:
    """Main orchestrator for clustering operations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.insights = []
        self.results = {}
        
    def run_complete_analysis(self) -> pd.DataFrame:
        """Run complete clustering analysis pipeline."""
        print("    - Engineering features...")
        fe = FeatureEngineering(self.df)
        self.df, feature_cols = fe.prepare_features()
        scaled_features = fe.get_scaled_features()
        
        print("    - Running K-Means clustering...")
        kmeans = KMeansClusterer(scaled_features)
        kmeans.find_optimal_k()
        kmeans_labels = kmeans.fit()
        self.df['kmeans_cluster'] = kmeans_labels
        self.results['kmeans'] = kmeans.results
        self.insights.extend(kmeans.insights)
        
        print("    - Running DBSCAN clustering...")
        dbscan = DBSCANClusterer(scaled_features)
        dbscan_labels = dbscan.fit()
        self.df['dbscan_cluster'] = dbscan_labels
        self.df['is_anomaly'] = dbscan_labels == -1
        self.results['dbscan'] = dbscan.results
        self.insights.extend(dbscan.insights)
        
        print("    - Running Hierarchical clustering...")
        hier = HierarchicalClusterer(scaled_features)
        hier.compute_linkage()
        hier_labels = hier.fit()
        self.df['hier_cluster'] = hier_labels
        self.results['hierarchical'] = hier.results
        self.insights.extend(hier.insights)
        
        print("    - Running GMM clustering...")
        gmm = GMMClusterer(scaled_features)
        gmm_labels = gmm.fit()
        self.df['gmm_cluster'] = gmm_labels
        self.results['gmm'] = gmm.results
        self.insights.extend(gmm.insights)
        
        print("    - Performing dimensionality reduction...")
        reducer = DimensionalityReducer(scaled_features)
        pca_coords, pca_results = reducer.pca(n_components=2)
        self.df['pca_1'] = pca_coords[:, 0]
        self.df['pca_2'] = pca_coords[:, 1]
        self.results['pca'] = pca_results
        self.insights.extend(reducer.insights)
        
        print("    - Profiling clusters...")
        profiler = ClusterProfiler(self.df, kmeans_labels, feature_cols)
        self.results['profiles'] = profiler.profile_all()
        self.df['price_segment'] = self.df['kmeans_cluster'].map(profiler.profiles.get('cluster_names', {}))
        self.insights.extend(profiler.insights)
        
        return self.df

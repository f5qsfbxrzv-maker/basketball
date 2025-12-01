"""
Bivariate Correlation Model and Derivative Markets
Joint modeling of spread and total with correlation structure
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class BivariateSpreadTotalModel:
    """
    Joint modeling of spread and total outcomes
    
    Models (Spread, Total) as bivariate normal with correlation:
    - Spread and total are correlated (high-scoring games often closer)
    - Enables pricing of derivative markets (parlays, teasers)
    - Better probability estimates for combined bets
    
    Uses Gaussian copula for flexibility:
    - Marginals can be non-normal (use empirical CDFs)
    - Dependence captured by correlation matrix
    """
    
    def __init__(self):
        self.spread_mean = 0.0
        self.spread_std = 12.0
        self.total_mean = 220.0
        self.total_std = 20.0
        self.correlation = 0.0
        self.fitted = False
        
    def fit(self, training_data: pd.DataFrame):
        """
        Fit bivariate model from historical games
        
        Args:
            training_data: DataFrame with columns:
                - home_score, away_score
                - spread (home - away margin)
                - total (home + away)
        """
        # Calculate spread and total if not present
        if 'spread' not in training_data.columns:
            training_data['spread'] = training_data['home_score'] - training_data['away_score']
        
        if 'total' not in training_data.columns:
            training_data['total'] = training_data['home_score'] + training_data['away_score']
        
        # Fit marginal distributions
        self.spread_mean = training_data['spread'].mean()
        self.spread_std = training_data['spread'].std()
        
        self.total_mean = training_data['total'].mean()
        self.total_std = training_data['total'].std()
        
        # Estimate correlation
        self.correlation = training_data[['spread', 'total']].corr().iloc[0, 1]
        
        self.fitted = True
        
        print(f"Bivariate model fitted:")
        print(f"  Spread: μ={self.spread_mean:.2f}, σ={self.spread_std:.2f}")
        print(f"  Total:  μ={self.total_mean:.2f}, σ={self.total_std:.2f}")
        print(f"  Correlation: ρ={self.correlation:.3f}")
    
    def predict_joint_probability(self,
                                  spread_line: float,
                                  total_line: float,
                                  expected_spread: float,
                                  expected_total: float,
                                  spread_std: Optional[float] = None,
                                  total_std: Optional[float] = None,
                                  correlation: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate joint probabilities for spread and total
        
        Args:
            spread_line: Spread betting line (home favorite if negative)
            total_line: Total betting line
            expected_spread: Model's expected spread
            expected_total: Model's expected total
            spread_std: Override spread standard deviation
            total_std: Override total standard deviation
            correlation: Override correlation coefficient
            
        Returns:
            Dict with probabilities for all 4 quadrants
        """
        if not self.fitted:
            print("Warning: Model not fitted. Using defaults.")
        
        # Use overrides if provided
        s_std = spread_std or self.spread_std
        t_std = total_std or self.total_std
        corr = correlation if correlation is not None else self.correlation
        
        # Bivariate normal CDF
        mean = np.array([expected_spread, expected_total])
        cov = np.array([
            [s_std**2, corr * s_std * t_std],
            [corr * s_std * t_std, t_std**2]
        ])
        
        rv = stats.multivariate_normal(mean=mean, cov=cov)
        
        # Calculate probabilities for each quadrant
        # P(Spread > line, Total > line)
        prob_cover_over = 1 - rv.cdf([spread_line, total_line])
        
        # P(Spread > line, Total <= line)
        # This requires integration - use Monte Carlo
        n_sim = 50000
        samples = rv.rvs(size=n_sim)
        
        cover_over = np.sum((samples[:, 0] > spread_line) & (samples[:, 1] > total_line)) / n_sim
        cover_under = np.sum((samples[:, 0] > spread_line) & (samples[:, 1] <= total_line)) / n_sim
        no_cover_over = np.sum((samples[:, 0] <= spread_line) & (samples[:, 1] > total_line)) / n_sim
        no_cover_under = np.sum((samples[:, 0] <= spread_line) & (samples[:, 1] <= total_line)) / n_sim
        
        return {
            'cover_and_over': cover_over,
            'cover_and_under': cover_under,
            'no_cover_and_over': no_cover_over,
            'no_cover_and_under': no_cover_under,
            'spread_cover_prob': cover_over + cover_under,
            'total_over_prob': cover_over + no_cover_over,
            'parlay_prob': cover_over,  # Both hit
            'expected_spread': expected_spread,
            'expected_total': expected_total,
            'correlation': corr
        }
    
    def price_parlay(self,
                    spread_line: float,
                    total_line: float,
                    expected_spread: float,
                    expected_total: float,
                    spread_odds: float,
                    total_odds: float) -> Dict[str, float]:
        """
        Price a spread + total parlay accounting for correlation
        
        Args:
            spread_line: Spread betting line
            total_line: Total betting line
            expected_spread: Model's expected spread
            expected_total: Model's expected total
            spread_odds: Decimal odds for spread
            total_odds: Decimal odds for total
            
        Returns:
            Dict with fair_parlay_odds, book_parlay_odds, edge
        """
        # Get joint probability
        joint_probs = self.predict_joint_probability(
            spread_line, total_line, expected_spread, expected_total
        )
        
        parlay_prob = joint_probs['parlay_prob']
        
        # Fair parlay odds
        fair_parlay_odds = 1.0 / parlay_prob if parlay_prob > 0 else 999
        
        # Book parlay odds (independent assumption)
        spread_prob = 1.0 / spread_odds
        total_prob = 1.0 / total_odds
        book_parlay_prob = spread_prob * total_prob
        book_parlay_odds = 1.0 / book_parlay_prob
        
        # Edge from correlation
        edge = parlay_prob - book_parlay_prob
        
        return {
            'true_parlay_prob': parlay_prob,
            'fair_parlay_odds': fair_parlay_odds,
            'book_parlay_odds': book_parlay_odds,
            'book_implied_prob': book_parlay_prob,
            'edge': edge,
            'correlation_benefit': edge / book_parlay_prob if book_parlay_prob > 0 else 0,
            'recommendation': 'BET' if edge > 0.02 else 'PASS'
        }
    
    def price_teaser(self,
                    spread_line: float,
                    total_line: float,
                    expected_spread: float,
                    expected_total: float,
                    teaser_points: float,
                    teaser_odds: float) -> Dict[str, float]:
        """
        Evaluate teaser value (e.g., 6-point teaser)
        
        Args:
            spread_line: Original spread line
            total_line: Original total line
            expected_spread: Model's expected spread
            expected_total: Model's expected total
            teaser_points: Points moved (e.g., 6.0)
            teaser_odds: Decimal odds offered (e.g., 1.526 for -190)
            
        Returns:
            Dict with teaser_edge, win_prob, fair_odds
        """
        # Adjust lines with teaser points
        # Spread: move toward home team
        # Total: can go either way, assume taking Over and getting points
        adj_spread_line = spread_line + teaser_points
        adj_total_line = total_line - teaser_points
        
        # Calculate win probability with adjusted lines
        joint_probs = self.predict_joint_probability(
            adj_spread_line, adj_total_line, expected_spread, expected_total
        )
        
        teaser_win_prob = joint_probs['parlay_prob']
        
        # Fair odds
        fair_teaser_odds = 1.0 / teaser_win_prob if teaser_win_prob > 0 else 999
        
        # Edge
        implied_prob = 1.0 / teaser_odds
        edge = teaser_win_prob - implied_prob
        
        return {
            'teaser_win_prob': teaser_win_prob,
            'fair_teaser_odds': fair_teaser_odds,
            'book_teaser_odds': teaser_odds,
            'book_implied_prob': implied_prob,
            'edge': edge,
            'ev_per_dollar': (teaser_win_prob * teaser_odds - 1.0),
            'recommendation': 'BET' if edge > 0.03 else 'PASS',
            'adjusted_spread': adj_spread_line,
            'adjusted_total': adj_total_line
        }
    
    def visualize_joint_distribution(self,
                                     expected_spread: float,
                                     expected_total: float,
                                     spread_line: float,
                                     total_line: float,
                                     save_path: Optional[str] = None):
        """
        Visualize bivariate distribution with betting lines
        
        Args:
            expected_spread: Model's expected spread
            expected_total: Model's expected total
            spread_line: Betting line for spread
            total_line: Betting line for total
            save_path: Path to save figure (optional)
        """
        # Generate grid
        spread_range = np.linspace(expected_spread - 3*self.spread_std,
                                   expected_spread + 3*self.spread_std, 100)
        total_range = np.linspace(expected_total - 3*self.total_std,
                                 expected_total + 3*self.total_std, 100)
        
        X, Y = np.meshgrid(spread_range, total_range)
        pos = np.dstack((X, Y))
        
        # Bivariate normal
        mean = np.array([expected_spread, expected_total])
        cov = np.array([
            [self.spread_std**2, self.correlation * self.spread_std * self.total_std],
            [self.correlation * self.spread_std * self.total_std, self.total_std**2]
        ])
        
        rv = stats.multivariate_normal(mean=mean, cov=cov)
        Z = rv.pdf(pos)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Probability Density')
        
        # Add betting lines
        plt.axvline(spread_line, color='red', linestyle='--', linewidth=2, label='Spread Line')
        plt.axhline(total_line, color='blue', linestyle='--', linewidth=2, label='Total Line')
        
        # Mark expected values
        plt.plot(expected_spread, expected_total, 'r*', markersize=20, label='Expected')
        
        plt.xlabel('Spread (Home - Away)', fontsize=12)
        plt.ylabel('Total Score', fontsize=12)
        plt.title(f'Joint Distribution (ρ={self.correlation:.3f})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


class CorrelationAnalyzer:
    """
    Analyze correlation patterns across different scenarios
    
    - Correlation varies by game context (pace, total level, etc.)
    - Identify when correlation is strongest
    - Adjust bivariate model accordingly
    """
    
    def __init__(self):
        self.correlations = {}
        
    def analyze_by_total_level(self, games_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how correlation changes with total score level
        
        Args:
            games_data: Historical games with spread and total
            
        Returns:
            DataFrame with total_bucket and correlation
        """
        # Add total if not present
        if 'total' not in games_data.columns:
            games_data['total'] = games_data['home_score'] + games_data['away_score']
        
        if 'spread' not in games_data.columns:
            games_data['spread'] = games_data['home_score'] - games_data['away_score']
        
        # Create total buckets
        games_data['total_bucket'] = pd.qcut(games_data['total'], 
                                             q=5, 
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Calculate correlation by bucket
        results = []
        for bucket in games_data['total_bucket'].unique():
            bucket_data = games_data[games_data['total_bucket'] == bucket]
            
            if len(bucket_data) > 10:
                corr = bucket_data[['spread', 'total']].corr().iloc[0, 1]
                
                results.append({
                    'total_bucket': bucket,
                    'correlation': corr,
                    'n_games': len(bucket_data),
                    'avg_total': bucket_data['total'].mean(),
                    'avg_spread_abs': bucket_data['spread'].abs().mean()
                })
        
        return pd.DataFrame(results)
    
    def analyze_by_pace(self, games_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlation by game pace
        
        Args:
            games_data: Historical games with pace, spread, total
            
        Returns:
            DataFrame with pace_bucket and correlation
        """
        if 'pace' not in games_data.columns:
            print("Warning: No pace data. Cannot analyze.")
            return pd.DataFrame()
        
        # Create pace buckets
        games_data['pace_bucket'] = pd.qcut(games_data['pace'],
                                           q=3,
                                           labels=['Slow', 'Medium', 'Fast'])
        
        results = []
        for bucket in games_data['pace_bucket'].unique():
            bucket_data = games_data[games_data['pace_bucket'] == bucket]
            
            if len(bucket_data) > 10:
                corr = bucket_data[['spread', 'total']].corr().iloc[0, 1]
                
                results.append({
                    'pace_bucket': bucket,
                    'correlation': corr,
                    'n_games': len(bucket_data),
                    'avg_pace': bucket_data['pace'].mean(),
                    'avg_total': bucket_data['total'].mean()
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    print("Bivariate Correlation Model Testing")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n_games = 1000
    
    # Generate correlated spread and total
    correlation = -0.15
    mean = [0, 220]
    cov = [[144, correlation * 12 * 20],
           [correlation * 12 * 20, 400]]
    
    samples = np.random.multivariate_normal(mean, cov, n_games)
    
    sample_data = pd.DataFrame({
        'spread': samples[:, 0],
        'total': samples[:, 1],
        'home_score': (samples[:, 1] + samples[:, 0]) / 2,
        'away_score': (samples[:, 1] - samples[:, 0]) / 2,
        'pace': np.random.normal(100, 5, n_games)
    })
    
    # Test bivariate model
    print("\n1. Testing Bivariate Model...")
    model = BivariateSpreadTotalModel()
    model.fit(sample_data)
    
    # Test joint probability
    joint_prob = model.predict_joint_probability(
        spread_line=-3.5,
        total_line=220,
        expected_spread=-5.0,
        expected_total=225
    )
    print("\nJoint probabilities:")
    for key, value in joint_prob.items():
        print(f"  {key}: {value:.4f}")
    
    # Test parlay pricing
    print("\n2. Testing Parlay Pricing...")
    parlay = model.price_parlay(
        spread_line=-3.5,
        total_line=220,
        expected_spread=-5.0,
        expected_total=225,
        spread_odds=1.91,  # -110
        total_odds=1.91
    )
    print("\nParlay analysis:")
    for key, value in parlay.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test teaser pricing
    print("\n3. Testing 6-Point Teaser...")
    teaser = model.price_teaser(
        spread_line=-3.5,
        total_line=220,
        expected_spread=-5.0,
        expected_total=225,
        teaser_points=6.0,
        teaser_odds=1.526  # -190
    )
    print("\nTeaser analysis:")
    for key, value in teaser.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test correlation analyzer
    print("\n4. Testing Correlation Analyzer...")
    analyzer = CorrelationAnalyzer()
    
    total_corr = analyzer.analyze_by_total_level(sample_data)
    print("\nCorrelation by total level:")
    print(total_corr)
    
    pace_corr = analyzer.analyze_by_pace(sample_data)
    print("\nCorrelation by pace:")
    print(pace_corr)
    
    print("\n✓ All tests completed successfully")

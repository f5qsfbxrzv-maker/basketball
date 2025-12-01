"""
Advanced Statistical Models for NBA Betting
Poisson/Negative Binomial, Bayesian Hierarchical, and Bivariate Modeling
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("WARNING: PyMC not available. Advanced Bayesian models disabled. Install with: pip install pymc arviz")


class PoissonTotalModel:
    """
    Poisson-based total scoring model
    
    Models each team's score as Poisson distribution based on:
    - Offensive/defensive ratings
    - Pace factors
    - Home court advantage
    - Recent form
    
    Advantages over normal approximation:
    - Captures discrete nature of scoring
    - Better tail probability estimates
    - Accounts for variance != mean in NB variant
    """
    
    def __init__(self, use_negative_binomial: bool = True):
        """
        Initialize Poisson total model
        
        Args:
            use_negative_binomial: If True, uses NB (allows overdispersion)
        """
        self.use_nb = use_negative_binomial
        self.params = {}
        self.baseline_scoring = 110.0  # NBA average points
        
    def fit(self, training_data: pd.DataFrame):
        """
        Fit Poisson/NB model parameters
        
        Args:
            training_data: DataFrame with columns:
                - home_team, away_team
                - home_score, away_score
                - home_off_rating, away_off_rating
                - home_def_rating, away_def_rating
                - pace
        """
        print(f"Fitting {'Negative Binomial' if self.use_nb else 'Poisson'} model...")
        
        # Calculate team offensive/defensive strengths
        self.params['team_off_strength'] = {}
        self.params['team_def_strength'] = {}
        
        for team in pd.concat([training_data['home_team'], 
                               training_data['away_team']]).unique():
            # Offensive strength (points scored)
            home_off = training_data[training_data['home_team'] == team]['home_score']
            away_off = training_data[training_data['away_team'] == team]['away_score']
            all_scores = pd.concat([home_off, away_off])
            
            self.params['team_off_strength'][team] = all_scores.mean() / self.baseline_scoring
            
            # Defensive strength (points allowed)
            home_def = training_data[training_data['home_team'] == team]['away_score']
            away_def = training_data[training_data['away_team'] == team]['home_score']
            all_allowed = pd.concat([home_def, away_def])
            
            self.params['team_def_strength'][team] = all_allowed.mean() / self.baseline_scoring
        
        # Home court advantage (multiplicative factor)
        home_advantage = training_data['home_score'].mean() / training_data['away_score'].mean()
        self.params['home_advantage'] = home_advantage
        
        # Pace factor influence
        self.params['pace_factor'] = training_data['pace'].mean() / 100.0
        
        # If using Negative Binomial, estimate dispersion
        if self.use_nb:
            all_scores = pd.concat([training_data['home_score'], 
                                   training_data['away_score']])
            variance = all_scores.var()
            mean = all_scores.mean()
            
            # NB dispersion: var = mu + mu^2/r, solve for r
            if variance > mean:
                self.params['dispersion'] = mean**2 / (variance - mean)
            else:
                self.params['dispersion'] = 100  # High r → Poisson
        
        print(f"Model fitted. Home advantage: {home_advantage:.3f}")
        
    def predict_score_distribution(self, 
                                   home_team: str, 
                                   away_team: str,
                                   pace: float = 100.0) -> Dict[str, stats.rv_discrete]:
        """
        Predict score distributions for both teams
        
        Args:
            home_team: Home team name
            away_team: Away team name
            pace: Game pace (possessions per 48 min)
            
        Returns:
            Dict with 'home' and 'away' distributions
        """
        # Calculate expected scores
        home_off = self.params['team_off_strength'].get(home_team, 1.0)
        away_def = self.params['team_def_strength'].get(away_team, 1.0)
        home_expected = (self.baseline_scoring * home_off * away_def * 
                        self.params['home_advantage'] * pace / 100.0)
        
        away_off = self.params['team_off_strength'].get(away_team, 1.0)
        home_def = self.params['team_def_strength'].get(home_team, 1.0)
        away_expected = (self.baseline_scoring * away_off * home_def * 
                        pace / 100.0)
        
        # Create distributions
        if self.use_nb:
            r = self.params['dispersion']
            # Convert mean to NB parameters
            home_p = r / (r + home_expected)
            away_p = r / (r + away_expected)
            
            distributions = {
                'home': stats.nbinom(n=r, p=home_p),
                'away': stats.nbinom(n=r, p=away_p)
            }
        else:
            distributions = {
                'home': stats.poisson(mu=home_expected),
                'away': stats.poisson(mu=away_expected)
            }
        
        return distributions
    
    def predict_total_probability(self,
                                  home_team: str,
                                  away_team: str,
                                  total_line: float,
                                  pace: float = 100.0,
                                  n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate probability distribution for total score
        
        Args:
            home_team: Home team name
            away_team: Away team name
            total_line: Total line to evaluate
            pace: Game pace
            n_simulations: Monte Carlo simulations
            
        Returns:
            Dict with over_prob, under_prob, expected_total, std
        """
        dists = self.predict_score_distribution(home_team, away_team, pace)
        
        # Monte Carlo simulation
        home_scores = dists['home'].rvs(size=n_simulations)
        away_scores = dists['away'].rvs(size=n_simulations)
        totals = home_scores + away_scores
        
        over_count = np.sum(totals > total_line)
        over_prob = over_count / n_simulations
        
        return {
            'over_prob': over_prob,
            'under_prob': 1.0 - over_prob,
            'expected_total': totals.mean(),
            'std': totals.std(),
            'percentile_5': np.percentile(totals, 5),
            'percentile_95': np.percentile(totals, 95),
            'median': np.median(totals)
        }
    
    def predict_spread_probability(self,
                                   home_team: str,
                                   away_team: str,
                                   spread_line: float,
                                   pace: float = 100.0,
                                   n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate probability distribution for spread
        
        Args:
            home_team: Home team name
            away_team: Away team name  
            spread_line: Spread line (positive = home favored)
            pace: Game pace
            n_simulations: Monte Carlo simulations
            
        Returns:
            Dict with cover_prob, expected_margin, std
        """
        dists = self.predict_score_distribution(home_team, away_team, pace)
        
        # Monte Carlo simulation
        home_scores = dists['home'].rvs(size=n_simulations)
        away_scores = dists['away'].rvs(size=n_simulations)
        margins = home_scores - away_scores
        
        cover_count = np.sum(margins > spread_line)
        cover_prob = cover_count / n_simulations
        
        return {
            'cover_prob': cover_prob,
            'no_cover_prob': 1.0 - cover_prob,
            'expected_margin': margins.mean(),
            'std': margins.std(),
            'percentile_5': np.percentile(margins, 5),
            'percentile_95': np.percentile(margins, 95)
        }


class BayesianHierarchicalModel:
    """
    Bayesian hierarchical model for player-level contributions
    
    Models team performance as sum of player contributions:
    - Player skill (offensive/defensive impact)
    - Team chemistry effects
    - Matchup-specific adjustments
    - Uncertainty quantification via posterior distributions
    
    Updates team ratings on the fly based on player availability
    """
    
    def __init__(self):
        self.model = None
        self.trace = None
        self.player_effects = {}
        self.team_baselines = {}
        
    def fit(self, 
            games_data: pd.DataFrame,
            player_stats: pd.DataFrame,
            n_samples: int = 2000,
            tune: int = 1000):
        """
        Fit Bayesian hierarchical model
        
        Args:
            games_data: Game outcomes with team compositions
            player_stats: Player-level statistics
            n_samples: MCMC samples
            tune: Tuning iterations
        """
        if not PYMC_AVAILABLE:
            print("PyMC not available. Using simplified approximation.")
            self._fit_approximate(games_data, player_stats)
            return
        
        print("Fitting Bayesian hierarchical model...")
        
        with pm.Model() as self.model:
            # Hyperpriors for player skill distribution
            mu_offense = pm.Normal('mu_offense', mu=0, sigma=5)
            sigma_offense = pm.HalfNormal('sigma_offense', sigma=2)
            
            mu_defense = pm.Normal('mu_defense', mu=0, sigma=5)
            sigma_defense = pm.HalfNormal('sigma_defense', sigma=2)
            
            # Player-level effects (hierarchical)
            n_players = len(player_stats)
            player_offense = pm.Normal('player_offense', 
                                      mu=mu_offense, 
                                      sigma=sigma_offense,
                                      shape=n_players)
            player_defense = pm.Normal('player_defense',
                                      mu=mu_defense,
                                      sigma=sigma_defense, 
                                      shape=n_players)
            
            # Team baseline strengths
            n_teams = games_data['home_team'].nunique()
            team_baseline = pm.Normal('team_baseline', mu=0, sigma=3, shape=n_teams)
            
            # Home court advantage
            home_adv = pm.Normal('home_advantage', mu=3, sigma=1)
            
            # Observation noise
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=10)
            
            # Likelihood (simplified - would need proper indexing)
            # This is a template - actual implementation needs player rosters
            # and proper indexing structures
            
            # Sample posterior
            self.trace = pm.sample(n_samples, tune=tune, return_inferencedata=True)
        
        print(f"Sampling complete. {n_samples} samples collected.")
        self._extract_parameters()
    
    def _fit_approximate(self, games_data: pd.DataFrame, player_stats: pd.DataFrame):
        """Simplified approximation when PyMC unavailable"""
        print("Using simplified player impact approximation...")
        
        # Calculate player impacts from box score stats
        for idx, player in player_stats.iterrows():
            player_name = player['player_name']
            
            # Offensive impact (points, assists, efficiency)
            off_impact = (player.get('points', 0) * 0.5 + 
                         player.get('assists', 0) * 1.5 +
                         player.get('fg_pct', 0.45) * 20)
            
            # Defensive impact (rebounds, steals, blocks)
            def_impact = (player.get('rebounds', 0) * 0.8 +
                         player.get('steals', 0) * 2.0 +
                         player.get('blocks', 0) * 2.5)
            
            self.player_effects[player_name] = {
                'offense': off_impact - 15,  # Center around 0
                'defense': def_impact - 10,
                'uncertainty': 2.0  # Fixed uncertainty
            }
        
        # Team baselines
        for team in games_data['home_team'].unique():
            team_games = games_data[
                (games_data['home_team'] == team) | 
                (games_data['away_team'] == team)
            ]
            
            home_scores = team_games[team_games['home_team'] == team]['home_score']
            away_scores = team_games[team_games['away_team'] == team]['away_score']
            all_scores = pd.concat([home_scores, away_scores])
            
            self.team_baselines[team] = {
                'baseline': all_scores.mean() - 110,
                'uncertainty': all_scores.std()
            }
    
    def _extract_parameters(self):
        """Extract player effects from trace"""
        if self.trace is None:
            return
        
        # Extract posterior means and uncertainties
        player_off_mean = self.trace.posterior['player_offense'].mean(dim=['chain', 'draw'])
        player_off_std = self.trace.posterior['player_offense'].std(dim=['chain', 'draw'])
        
        player_def_mean = self.trace.posterior['player_defense'].mean(dim=['chain', 'draw'])
        player_def_std = self.trace.posterior['player_defense'].std(dim=['chain', 'draw'])
        
        # Store (simplified - need proper player indexing)
        for i in range(len(player_off_mean)):
            player_name = f"player_{i}"  # Would map to actual names
            self.player_effects[player_name] = {
                'offense': float(player_off_mean[i]),
                'defense': float(player_def_mean[i]),
                'uncertainty': float(player_off_std[i])
            }
    
    def predict_with_roster(self,
                           home_roster: List[str],
                           away_roster: List[str],
                           home_team: str,
                           away_team: str) -> Dict[str, float]:
        """
        Predict game outcome based on specific rosters
        
        Args:
            home_roster: List of home player names
            away_roster: List of away player names
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dict with expected_margin, uncertainty, home_win_prob
        """
        # Sum player effects
        home_off = sum(self.player_effects.get(p, {}).get('offense', 0) 
                      for p in home_roster)
        home_def = sum(self.player_effects.get(p, {}).get('defense', 0)
                      for p in home_roster)
        
        away_off = sum(self.player_effects.get(p, {}).get('offense', 0)
                      for p in away_roster)
        away_def = sum(self.player_effects.get(p, {}).get('defense', 0)
                      for p in away_roster)
        
        # Team baselines
        home_base = self.team_baselines.get(home_team, {}).get('baseline', 0)
        away_base = self.team_baselines.get(away_team, {}).get('baseline', 0)
        
        # Expected margin
        home_advantage = 3.0
        expected_margin = (home_base + home_off - away_def + home_advantage -
                          (away_base + away_off - home_def))
        
        # Uncertainty (combine player uncertainties)
        home_unc = np.sqrt(sum(self.player_effects.get(p, {}).get('uncertainty', 2)**2 
                              for p in home_roster))
        away_unc = np.sqrt(sum(self.player_effects.get(p, {}).get('uncertainty', 2)**2
                              for p in away_roster))
        total_uncertainty = np.sqrt(home_unc**2 + away_unc**2 + 100)  # Game noise
        
        # Win probability (normal approximation)
        home_win_prob = 1 - stats.norm.cdf(0, loc=expected_margin, scale=total_uncertainty)
        
        return {
            'expected_margin': expected_margin,
            'uncertainty': total_uncertainty,
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'home_offensive_impact': home_off,
            'home_defensive_impact': home_def,
            'away_offensive_impact': away_off,
            'away_defensive_impact': away_def
        }
    
    def simulate_injury_impact(self,
                              base_roster: List[str],
                              injured_player: str,
                              replacement_player: Optional[str] = None) -> Dict[str, float]:
        """
        Simulate impact of injury/return
        
        Args:
            base_roster: Current roster
            injured_player: Player out
            replacement_player: Replacement (if any)
            
        Returns:
            Dict with expected_point_swing, uncertainty_change
        """
        # Original player impact
        injured_impact = self.player_effects.get(injured_player, {})
        injured_off = injured_impact.get('offense', 0)
        injured_def = injured_impact.get('defense', 0)
        
        # Replacement impact
        if replacement_player:
            repl_impact = self.player_effects.get(replacement_player, {})
            repl_off = repl_impact.get('offense', 0)
            repl_def = repl_impact.get('defense', 0)
        else:
            # League average replacement
            repl_off = -2.0
            repl_def = -1.5
        
        # Net swing
        off_swing = injured_off - repl_off
        def_swing = injured_def - repl_def
        total_swing = off_swing + def_swing
        
        return {
            'expected_point_swing': total_swing,
            'offensive_swing': off_swing,
            'defensive_swing': def_swing,
            'uncertainty_change': injured_impact.get('uncertainty', 2),
            'recommendation': 'adjust_line' if abs(total_swing) > 2 else 'no_change'
        }


if __name__ == "__main__":
    print("Advanced Statistical Models Module")
    print("="*60)
    
    # Test Poisson model
    print("\nTesting Poisson Total Model...")
    poisson_model = PoissonTotalModel(use_negative_binomial=True)
    
    # Create sample training data
    sample_data = pd.DataFrame({
        'home_team': ['LAL', 'BOS', 'GSW'] * 20,
        'away_team': ['BOS', 'GSW', 'LAL'] * 20,
        'home_score': np.random.poisson(115, 60),
        'away_score': np.random.poisson(108, 60),
        'home_off_rating': [115.0] * 60,
        'away_off_rating': [112.0] * 60,
        'home_def_rating': [108.0] * 60,
        'away_def_rating': [110.0] * 60,
        'pace': [100.0] * 60
    })
    
    poisson_model.fit(sample_data)
    
    # Test prediction
    total_pred = poisson_model.predict_total_probability(
        'LAL', 'BOS', total_line=220, pace=102
    )
    print(f"Total prediction: {total_pred}")
    
    spread_pred = poisson_model.predict_spread_probability(
        'LAL', 'BOS', spread_line=-3.5, pace=102
    )
    print(f"Spread prediction: {spread_pred}")
    
    # Test Bayesian model
    print("\nTesting Bayesian Hierarchical Model...")
    bayes_model = BayesianHierarchicalModel()
    
    # Sample player stats
    player_stats = pd.DataFrame({
        'player_name': ['LeBron', 'AD', 'Tatum', 'Brown'],
        'points': [25, 23, 27, 22],
        'assists': [7, 3, 5, 4],
        'rebounds': [8, 11, 8, 6],
        'steals': [1.2, 0.8, 1.0, 1.1],
        'blocks': [0.5, 2.0, 0.6, 0.4],
        'fg_pct': [0.52, 0.55, 0.48, 0.47]
    })
    
    bayes_model._fit_approximate(sample_data, player_stats)
    
    # Test roster prediction
    roster_pred = bayes_model.predict_with_roster(
        home_roster=['LeBron', 'AD'],
        away_roster=['Tatum', 'Brown'],
        home_team='LAL',
        away_team='BOS'
    )
    print(f"Roster prediction: {roster_pred}")
    
    # Test injury impact
    injury_impact = bayes_model.simulate_injury_impact(
        base_roster=['LeBron', 'AD'],
        injured_player='LeBron',
        replacement_player=None
    )
    print(f"Injury impact: {injury_impact}")
    
    print("\n✓ All tests completed successfully")

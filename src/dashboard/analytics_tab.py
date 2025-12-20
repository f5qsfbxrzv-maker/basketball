"""
Comprehensive Analytics Tab - Unified Results & Predictions Analysis
Replaces Trial 1306 Metrics + Legacy Performance with robust filtering and insights
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTableWidget, 
    QTableWidgetItem, QLabel, QGroupBox, QPushButton, QComboBox,
    QDateEdit, QCheckBox, QHeaderView, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QDate
from PyQt6.QtGui import QFont, QColor


class AnalyticsTab(QWidget):
    """Main analytics tab with sub-tabs for comprehensive analysis"""
    
    def __init__(self, predictor, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.db_path = 'data/live/nba_betting_data.db'
        self.init_ui()
        
        # Auto-load data on startup
        self.load_all_data()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("📊 COMPREHENSIVE ANALYTICS & RESULTS")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("padding: 10px; background-color: #1e3a5f; color: #FFD700;")
        layout.addWidget(header)
        
        # Sub-tabs
        self.tabs = QTabWidget()
        
        # 1. Overview Dashboard
        self.overview_tab = OverviewTab(self.db_path, self)
        self.tabs.addTab(self.overview_tab, "📈 Overview")
        
        # 2. Detailed Results (filterable)
        self.results_tab = ResultsTab(self.db_path, self)
        self.tabs.addTab(self.results_tab, "🎯 Results")
        
        # 3. Predictions Log (all predictions with outcomes)
        self.predictions_tab = PredictionsLogTab(self.db_path, self)
        self.tabs.addTab(self.predictions_tab, "📋 Predictions Log")
        
        # 4. Team Analysis
        self.team_analysis_tab = TeamAnalysisTab(self.db_path, self)
        self.tabs.addTab(self.team_analysis_tab, "🏀 Team Analysis")
        
        # 5. Edge Analysis
        self.edge_analysis_tab = EdgeAnalysisTab(self.db_path, self)
        self.tabs.addTab(self.edge_analysis_tab, "📊 Edge Analysis")
        
        # 6. Calibration & Accuracy
        self.calibration_tab = CalibrationTab(self.db_path, self)
        self.tabs.addTab(self.calibration_tab, "🎲 Calibration")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def load_all_data(self):
        """Load data into all tabs automatically"""
        self.overview_tab.refresh_data()
        self.results_tab.load_results()
        self.predictions_tab.load_predictions()
        self.team_analysis_tab.refresh_analysis()
        self.edge_analysis_tab.refresh_analysis()
        self.calibration_tab.refresh_calibration()


class OverviewTab(QWidget):
    """High-level performance dashboard with key metrics"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh All Metrics")
        refresh_btn.clicked.connect(self.refresh_data)
        refresh_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        layout.addWidget(refresh_btn)
        
        # Overall Performance Group
        perf_group = QGroupBox("🎯 Overall Performance")
        perf_layout = QVBoxLayout()
        
        self.overall_stats_label = QLabel("Loading...")
        self.overall_stats_label.setTextFormat(Qt.TextFormat.RichText)
        perf_layout.addWidget(self.overall_stats_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Split Performance (Home/Away, Fav/Dog)
        split_group = QGroupBox("📊 Performance Splits")
        split_layout = QVBoxLayout()
        
        self.split_stats_label = QLabel("Loading...")
        self.split_stats_label.setTextFormat(Qt.TextFormat.RichText)
        split_layout.addWidget(self.split_stats_label)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        # Recent Trends (Last 7/30 days)
        trends_group = QGroupBox("📅 Recent Trends")
        trends_layout = QVBoxLayout()
        
        self.trends_label = QLabel("Loading...")
        self.trends_label.setTextFormat(Qt.TextFormat.RichText)
        trends_layout.addWidget(self.trends_label)
        
        trends_group.setLayout(trends_layout)
        layout.addWidget(trends_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def refresh_data(self):
        """Load and display all overview metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall stats
            overall_df = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome IS NULL THEN 1 ELSE 0 END) as pending,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(stake_amount) as total_staked,
                    SUM(COALESCE(profit_loss, 0)) as total_profit,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi,
                    AVG(edge) as avg_edge,
                    AVG(model_probability) as avg_model_prob
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
            """, conn)
            
            if not overall_df.empty and overall_df['total_bets'].iloc[0] > 0:
                stats = overall_df.iloc[0]
                win_rate = stats['win_rate'] * 100 if pd.notna(stats['win_rate']) else 0
                roi = stats['roi'] * 100 if pd.notna(stats['roi']) else 0
                roi_color = '#90EE90' if roi > 0 else '#FF6B6B'
                
                self.overall_stats_label.setText(f"""
                <h3>All-Time Results</h3>
                <table style='width: 100%; font-size: 14px;'>
                <tr><td><b>Total Bets:</b></td><td>{int(stats['total_bets'])}</td></tr>
                <tr><td><b>Record:</b></td><td>{int(stats['wins'])}-{int(stats['losses'])}-{int(stats['pending'])}</td></tr>
                <tr><td><b>Win Rate:</b></td><td style='color: {roi_color};'><b>{win_rate:.1f}%</b></td></tr>
                <tr><td><b>Total Staked:</b></td><td>${stats['total_staked']:.2f}</td></tr>
                <tr><td><b>Total Profit:</b></td><td style='color: {roi_color};'><b>${stats['total_profit']:.2f}</b></td></tr>
                <tr><td><b>ROI:</b></td><td style='color: {roi_color}; font-size: 18px;'><b>{roi:.1f}%</b></td></tr>
                <tr><td><b>Avg Edge:</b></td><td>{stats['avg_edge']*100:.2f}%</td></tr>
                <tr><td><b>Avg Model Prob:</b></td><td>{stats['avg_model_prob']*100:.1f}%</td></tr>
                </table>
                """)
            else:
                self.overall_stats_label.setText("<h3>No graded bets yet</h3>")
            
            # Split performance
            home_away = pd.read_sql_query("""
                SELECT 
                    CASE WHEN predicted_winner = home_team THEN 'Home' ELSE 'Away' END as side,
                    COUNT(*) as bets,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
                GROUP BY side
            """, conn)
            
            fav_dog = pd.read_sql_query("""
                SELECT 
                    CASE WHEN model_probability > 0.5 THEN 'Favorite' ELSE 'Underdog' END as type,
                    COUNT(*) as bets,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
                GROUP BY type
            """, conn)
            
            split_html = "<h3>Performance Breakdown</h3><table style='width: 100%; font-size: 14px;'>"
            split_html += "<tr><th>Category</th><th>Bets</th><th>Win Rate</th><th>ROI</th></tr>"
            
            for _, row in home_away.iterrows():
                wr = row['win_rate'] * 100 if pd.notna(row['win_rate']) else 0
                roi_val = row['roi'] * 100 if pd.notna(row['roi']) else 0
                roi_color = '#90EE90' if roi_val > 0 else '#FF6B6B'
                split_html += f"<tr><td><b>{row['side']}</b></td><td>{int(row['bets'])}</td><td>{wr:.1f}%</td><td style='color: {roi_color};'><b>{roi_val:.1f}%</b></td></tr>"
            
            for _, row in fav_dog.iterrows():
                wr = row['win_rate'] * 100 if pd.notna(row['win_rate']) else 0
                roi_val = row['roi'] * 100 if pd.notna(row['roi']) else 0
                roi_color = '#90EE90' if roi_val > 0 else '#FF6B6B'
                split_html += f"<tr><td><b>{row['type']}</b></td><td>{int(row['bets'])}</td><td>{wr:.1f}%</td><td style='color: {roi_color};'><b>{roi_val:.1f}%</b></td></tr>"
            
            split_html += "</table>"
            self.split_stats_label.setText(split_html)
            
            # Recent trends
            recent_7 = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as bets,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi
                FROM trial1306_bets
                WHERE outcome IS NOT NULL 
                AND bet_date >= datetime('now', '-7 days')
            """, conn)
            
            recent_30 = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as bets,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi
                FROM trial1306_bets
                WHERE outcome IS NOT NULL 
                AND bet_date >= datetime('now', '-30 days')
            """, conn)
            
            trends_html = "<h3>Recent Performance</h3><table style='width: 100%; font-size: 14px;'>"
            trends_html += "<tr><th>Period</th><th>Bets</th><th>Win Rate</th><th>ROI</th></tr>"
            
            for period, df in [("Last 7 Days", recent_7), ("Last 30 Days", recent_30)]:
                if not df.empty and df['bets'].iloc[0] > 0:
                    wr = df['win_rate'].iloc[0] * 100 if pd.notna(df['win_rate'].iloc[0]) else 0
                    roi_val = df['roi'].iloc[0] * 100 if pd.notna(df['roi'].iloc[0]) else 0
                    roi_color = '#90EE90' if roi_val > 0 else '#FF6B6B'
                    trends_html += f"<tr><td><b>{period}</b></td><td>{int(df['bets'].iloc[0])}</td><td>{wr:.1f}%</td><td style='color: {roi_color};'><b>{roi_val:.1f}%</b></td></tr>"
                else:
                    trends_html += f"<tr><td><b>{period}</b></td><td colspan='3'>No bets</td></tr>"
            
            trends_html += "</table>"
            self.trends_label.setText(trends_html)
            
            conn.close()
            
        except Exception as e:
            print(f"[ERROR] Loading overview: {e}")
            import traceback
            traceback.print_exc()
            self.overall_stats_label.setText(f"<h3>Error loading data</h3><p>{str(e)}</p>")


class ResultsTab(QWidget):
    """Detailed bet results with comprehensive filtering"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Filters
        filter_group = QGroupBox("🔍 Filters")
        filter_layout = QHBoxLayout()
        
        # Date range
        filter_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addDays(-30))
        self.date_from.setCalendarPopup(True)
        filter_layout.addWidget(self.date_from)
        
        filter_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate().addDays(1))
        self.date_to.setCalendarPopup(True)
        filter_layout.addWidget(self.date_to)
        
        # Side filter (Home/Away/All)
        filter_layout.addWidget(QLabel("Side:"))
        self.side_combo = QComboBox()
        self.side_combo.addItems(["All", "Home", "Away"])
        filter_layout.addWidget(self.side_combo)
        
        # Favorite/Dog filter
        filter_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["All", "Favorite", "Underdog"])
        filter_layout.addWidget(self.type_combo)
        
        # Team filter
        filter_layout.addWidget(QLabel("Team:"))
        self.team_combo = QComboBox()
        self.team_combo.addItem("All Teams")
        filter_layout.addWidget(self.team_combo)
        
        # Edge range
        filter_layout.addWidget(QLabel("Min Edge %:"))
        self.min_edge_spin = QSpinBox()
        self.min_edge_spin.setRange(0, 50)
        self.min_edge_spin.setValue(0)
        filter_layout.addWidget(self.min_edge_spin)
        
        # Apply button
        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.load_results)
        apply_btn.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        filter_layout.addWidget(apply_btn)
        
        filter_layout.addStretch()
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)
        
        # Summary stats for filtered results
        self.summary_label = QLabel("No data")
        self.summary_label.setStyleSheet("padding: 10px; background-color: #2c3e50; color: white; font-size: 14px;")
        layout.addWidget(self.summary_label)
        
        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "Date", "Matchup", "Pick", "Side", "Type",
            "Model %", "Edge %", "Odds", "Stake",
            "Result", "P/L", "ROI %", "Margin"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
        
        # Populate team dropdown
        self.populate_teams()
    
    def populate_teams(self):
        """Load all unique teams from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            teams = pd.read_sql_query("""
                SELECT DISTINCT home_team as team FROM trial1306_bets
                UNION
                SELECT DISTINCT away_team as team FROM trial1306_bets
                ORDER BY team
            """, conn)
            conn.close()
            
            for team in teams['team']:
                self.team_combo.addItem(team)
        except:
            pass
    
    def load_results(self):
        """Load results with current filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Build query with filters
            query = """
                SELECT 
                    bet_date,
                    game_date,
                    home_team,
                    away_team,
                    predicted_winner,
                    model_probability,
                    edge,
                    market_odds,
                    stake_amount,
                    outcome,
                    profit_loss,
                    home_score,
                    away_score,
                    actual_winner
                FROM trial1306_bets
                WHERE 1=1
            """
            params = []
            
            # Date filter
            date_from = self.date_from.date().toString("yyyy-MM-dd")
            date_to = self.date_to.date().toString("yyyy-MM-dd")
            query += " AND game_date BETWEEN ? AND ?"
            params.extend([date_from, date_to])
            
            # Side filter
            side = self.side_combo.currentText()
            if side == "Home":
                query += " AND predicted_winner = home_team"
            elif side == "Away":
                query += " AND predicted_winner = away_team"
            
            # Type filter
            type_filter = self.type_combo.currentText()
            if type_filter == "Favorite":
                query += " AND model_probability > 0.5"
            elif type_filter == "Underdog":
                query += " AND model_probability <= 0.5"
            
            # Team filter
            team = self.team_combo.currentText()
            if team != "All Teams":
                query += " AND (home_team = ? OR away_team = ?)"
                params.extend([team, team])
            
            # Edge filter
            min_edge = self.min_edge_spin.value() / 100
            query += " AND edge >= ?"
            params.append(min_edge)
            
            query += " ORDER BY game_date DESC, bet_date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            # Update summary
            if not df.empty:
                graded = df[df['outcome'].notna()]
                if not graded.empty:
                    wins = (graded['outcome'] == 'win').sum()
                    losses = (graded['outcome'] == 'loss').sum()
                    total = len(graded)
                    win_rate = wins / total * 100 if total > 0 else 0
                    total_profit = graded['profit_loss'].sum()
                    total_staked = graded['stake_amount'].sum()
                    roi = total_profit / total_staked * 100 if total_staked > 0 else 0
                    roi_color = '#90EE90' if roi > 0 else '#FF6B6B'
                    
                    self.summary_label.setText(
                        f"<b>Filtered Results:</b> {total} bets | "
                        f"Record: {wins}-{losses} | "
                        f"Win Rate: {win_rate:.1f}% | "
                        f"Profit: ${total_profit:.2f} | "
                        f"<span style='color: {roi_color}; font-size: 16px;'><b>ROI: {roi:.1f}%</b></span>"
                    )
                else:
                    self.summary_label.setText(f"<b>{len(df)} bets found, none graded yet</b>")
            else:
                self.summary_label.setText("<b>No bets match filters</b>")
            
            # Populate table
            self.table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.table.setItem(i, 0, QTableWidgetItem(row['game_date']))
                self.table.setItem(i, 1, QTableWidgetItem(f"{row['away_team']} @ {row['home_team']}"))
                self.table.setItem(i, 2, QTableWidgetItem(row['predicted_winner']))
                
                # Side
                side = "Home" if row['predicted_winner'] == row['home_team'] else "Away"
                self.table.setItem(i, 3, QTableWidgetItem(side))
                
                # Type
                bet_type = "Fav" if row['model_probability'] > 0.5 else "Dog"
                self.table.setItem(i, 4, QTableWidgetItem(bet_type))
                
                self.table.setItem(i, 5, QTableWidgetItem(f"{row['model_probability']*100:.1f}"))
                self.table.setItem(i, 6, QTableWidgetItem(f"{row['edge']*100:.2f}"))
                self.table.setItem(i, 7, QTableWidgetItem(f"{row['market_odds']:.0f}"))
                self.table.setItem(i, 8, QTableWidgetItem(f"${row['stake_amount']:.2f}"))
                
                # Result
                if pd.notna(row['outcome']):
                    result_item = QTableWidgetItem(row['outcome'].upper())
                    if row['outcome'] == 'win':
                        result_item.setForeground(QColor('#90EE90'))
                    else:
                        result_item.setForeground(QColor('#FF6B6B'))
                    self.table.setItem(i, 9, result_item)
                    
                    # P/L
                    pl_item = QTableWidgetItem(f"${row['profit_loss']:.2f}")
                    pl_item.setForeground(QColor('#90EE90' if row['profit_loss'] > 0 else '#FF6B6B'))
                    self.table.setItem(i, 10, pl_item)
                    
                    # ROI
                    roi = row['profit_loss'] / row['stake_amount'] * 100
                    roi_item = QTableWidgetItem(f"{roi:.1f}")
                    roi_item.setForeground(QColor('#90EE90' if roi > 0 else '#FF6B6B'))
                    self.table.setItem(i, 11, roi_item)
                    
                    # Margin
                    if pd.notna(row['home_score']) and pd.notna(row['away_score']):
                        margin = abs(row['home_score'] - row['away_score'])
                        self.table.setItem(i, 12, QTableWidgetItem(f"{margin:.0f}"))
                else:
                    self.table.setItem(i, 9, QTableWidgetItem("PENDING"))
                    self.table.setItem(i, 10, QTableWidgetItem("-"))
                    self.table.setItem(i, 11, QTableWidgetItem("-"))
                    self.table.setItem(i, 12, QTableWidgetItem("-"))
            
        except Exception as e:
            print(f"[ERROR] Loading results: {e}")
            import traceback
            traceback.print_exc()
            self.summary_label.setText(f"<b>Error:</b> {str(e)}")


class PredictionsLogTab(QWidget):
    """All predictions (bet or not) with outcomes and accuracy"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Controls
        controls = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_predictions)
        controls.addWidget(refresh_btn)
        
        controls.addWidget(QLabel("Days back:"))
        self.days_spin = QSpinBox()
        self.days_spin.setRange(1, 365)
        self.days_spin.setValue(30)
        controls.addWidget(self.days_spin)
        
        self.only_graded_check = QCheckBox("Only Graded")
        self.only_graded_check.setChecked(True)
        controls.addWidget(self.only_graded_check)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Accuracy summary
        self.accuracy_label = QLabel("No data")
        self.accuracy_label.setStyleSheet("padding: 10px; background-color: #2c3e50; color: white; font-size: 14px;")
        layout.addWidget(self.accuracy_label)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(12)
        self.table.setHorizontalHeaderLabels([
            "Date", "Matchup", "Model Pick", "Model %",
            "Opening Odds", "Opening Edge %", "Closing Odds", "Closing Edge %",
            "Qualified?", "Winner", "Correct?", "Margin"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def load_predictions(self):
        """Load all predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            days_back = self.days_spin.value()
            only_graded = self.only_graded_check.isChecked()
            
            query = """
                SELECT 
                    game_date,
                    home_team,
                    away_team,
                    model_home_prob,
                    model_away_prob,
                    opening_home_ml,
                    opening_away_ml,
                    opening_home_edge,
                    opening_away_edge,
                    closing_home_ml,
                    closing_away_ml,
                    closing_home_edge,
                    closing_away_edge,
                    qualified_bet,
                    best_pick,
                    actual_winner,
                    home_score,
                    away_score
                FROM daily_predictions
                WHERE game_date >= date('now', ?)
            """
            
            params = [f'-{days_back} days']
            
            if only_graded:
                query += " AND actual_winner IS NOT NULL"
            
            query += " ORDER BY game_date DESC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                self.accuracy_label.setText("<b>No predictions found</b>")
                self.table.setRowCount(0)
                return
            
            # Calculate accuracy
            graded = df[df['actual_winner'].notna()].copy()
            if not graded.empty:
                # Determine model pick
                graded['model_pick'] = graded.apply(
                    lambda r: r['home_team'] if r['model_home_prob'] > r['model_away_prob'] else r['away_team'],
                    axis=1
                )
                graded['correct'] = graded['model_pick'] == graded['actual_winner']
                
                accuracy = graded['correct'].mean() * 100
                total = len(graded)
                correct = graded['correct'].sum()
                
                # Margin accuracy
                graded['predicted_margin'] = abs(
                    (graded['model_home_prob'] - graded['model_away_prob']) * 100
                )
                graded['actual_margin'] = abs(graded['home_score'] - graded['away_score'])
                graded['margin_error'] = abs(graded['predicted_margin'] - graded['actual_margin'])
                avg_margin_error = graded['margin_error'].mean()
                
                self.accuracy_label.setText(
                    f"<b>Accuracy:</b> {correct}/{total} ({accuracy:.1f}%) | "
                    f"<b>Avg Margin Error:</b> {avg_margin_error:.1f} pts | "
                    f"<b>Qualified Bets:</b> {graded['qualified_bet'].sum()} of {total}"
                )
            else:
                self.accuracy_label.setText(f"<b>{len(df)} predictions, none graded yet</b>")
            
            # Populate table
            self.table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.table.setItem(i, 0, QTableWidgetItem(row['game_date']))
                self.table.setItem(i, 1, QTableWidgetItem(f"{row['away_team']} @ {row['home_team']}"))
                
                # Model pick
                model_pick = row['home_team'] if row['model_home_prob'] > row['model_away_prob'] else row['away_team']
                model_prob = max(row['model_home_prob'], row['model_away_prob']) * 100
                self.table.setItem(i, 2, QTableWidgetItem(model_pick))
                self.table.setItem(i, 3, QTableWidgetItem(f"{model_prob:.1f}"))
                
                # Opening odds & edge
                if pd.notna(row['opening_home_ml']):
                    opening_odds = f"{row['opening_home_ml']}/{row['opening_away_ml']}"
                    self.table.setItem(i, 4, QTableWidgetItem(opening_odds))
                    
                    best_opening_edge = max(
                        abs(row['opening_home_edge']) if pd.notna(row['opening_home_edge']) else 0,
                        abs(row['opening_away_edge']) if pd.notna(row['opening_away_edge']) else 0
                    ) * 100
                    self.table.setItem(i, 5, QTableWidgetItem(f"{best_opening_edge:.2f}"))
                else:
                    self.table.setItem(i, 4, QTableWidgetItem("N/A"))
                    self.table.setItem(i, 5, QTableWidgetItem("N/A"))
                
                # Closing odds & edge
                if pd.notna(row['closing_home_ml']):
                    closing_odds = f"{row['closing_home_ml']}/{row['closing_away_ml']}"
                    self.table.setItem(i, 6, QTableWidgetItem(closing_odds))
                    
                    best_closing_edge = max(
                        abs(row['closing_home_edge']) if pd.notna(row['closing_home_edge']) else 0,
                        abs(row['closing_away_edge']) if pd.notna(row['closing_away_edge']) else 0
                    ) * 100
                    self.table.setItem(i, 7, QTableWidgetItem(f"{best_closing_edge:.2f}"))
                else:
                    self.table.setItem(i, 6, QTableWidgetItem("N/A"))
                    self.table.setItem(i, 7, QTableWidgetItem("N/A"))
                
                # Qualified
                qualified_item = QTableWidgetItem("YES" if row['qualified_bet'] else "NO")
                if row['qualified_bet']:
                    qualified_item.setForeground(QColor('#90EE90'))
                self.table.setItem(i, 8, qualified_item)
                
                # Winner & correctness
                if pd.notna(row['actual_winner']):
                    self.table.setItem(i, 9, QTableWidgetItem(row['actual_winner']))
                    
                    correct = model_pick == row['actual_winner']
                    correct_item = QTableWidgetItem("✓" if correct else "✗")
                    correct_item.setForeground(QColor('#90EE90' if correct else '#FF6B6B'))
                    self.table.setItem(i, 10, correct_item)
                    
                    # Margin
                    if pd.notna(row['home_score']) and pd.notna(row['away_score']):
                        margin = abs(row['home_score'] - row['away_score'])
                        self.table.setItem(i, 11, QTableWidgetItem(f"{margin:.0f}"))
                else:
                    self.table.setItem(i, 9, QTableWidgetItem("PENDING"))
                    self.table.setItem(i, 10, QTableWidgetItem("-"))
                    self.table.setItem(i, 11, QTableWidgetItem("-"))
            
        except Exception as e:
            print(f"[ERROR] Loading predictions: {e}")
            import traceback
            traceback.print_exc()


class TeamAnalysisTab(QWidget):
    """Performance breakdown by team"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Team Stats")
        refresh_btn.clicked.connect(self.refresh_analysis)
        layout.addWidget(refresh_btn)
        
        # Team table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Team", "Total Bets", "Wins", "Losses", "Win Rate %",
            "Total Staked", "Total Profit", "ROI %", "Avg Edge %", "Avg Prob %"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def refresh_analysis(self):
        """Load team-by-team stats"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get stats for each team (both when betting on them and against)
            df = pd.read_sql_query("""
                SELECT 
                    predicted_winner as team,
                    COUNT(*) as total_bets,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                    SUM(stake_amount) as total_staked,
                    SUM(COALESCE(profit_loss, 0)) as total_profit,
                    SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi,
                    AVG(edge) as avg_edge,
                    AVG(model_probability) as avg_prob
                FROM trial1306_bets
                WHERE outcome IS NOT NULL
                GROUP BY predicted_winner
                ORDER BY roi DESC, win_rate DESC
            """, conn)
            
            conn.close()
            
            # Populate table
            self.table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.table.setItem(i, 0, QTableWidgetItem(row['team']))
                self.table.setItem(i, 1, QTableWidgetItem(str(int(row['total_bets']))))
                self.table.setItem(i, 2, QTableWidgetItem(str(int(row['wins']))))
                self.table.setItem(i, 3, QTableWidgetItem(str(int(row['losses']))))
                
                win_rate = row['win_rate'] * 100
                wr_item = QTableWidgetItem(f"{win_rate:.1f}")
                wr_item.setForeground(QColor('#90EE90' if win_rate > 50 else 'white'))
                self.table.setItem(i, 4, wr_item)
                
                self.table.setItem(i, 5, QTableWidgetItem(f"${row['total_staked']:.2f}"))
                
                profit_item = QTableWidgetItem(f"${row['total_profit']:.2f}")
                profit_item.setForeground(QColor('#90EE90' if row['total_profit'] > 0 else '#FF6B6B'))
                self.table.setItem(i, 6, profit_item)
                
                roi = row['roi'] * 100
                roi_item = QTableWidgetItem(f"{roi:.1f}")
                roi_item.setForeground(QColor('#90EE90' if roi > 0 else '#FF6B6B'))
                self.table.setItem(i, 7, roi_item)
                
                self.table.setItem(i, 8, QTableWidgetItem(f"{row['avg_edge']*100:.2f}"))
                self.table.setItem(i, 9, QTableWidgetItem(f"{row['avg_prob']*100:.1f}"))
            
        except Exception as e:
            print(f"[ERROR] Loading team analysis: {e}")
            import traceback
            traceback.print_exc()


class EdgeAnalysisTab(QWidget):
    """Performance by edge buckets"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Edge Analysis")
        refresh_btn.clicked.connect(self.refresh_analysis)
        layout.addWidget(refresh_btn)
        
        # Edge buckets table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Edge Range", "Bets", "Wins", "Losses", "Win Rate %",
            "Total Profit", "ROI %", "Avg Stake"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def refresh_analysis(self):
        """Analyze performance by edge buckets"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Define edge buckets
            buckets = [
                ("0-2%", 0, 0.02),
                ("2-4%", 0.02, 0.04),
                ("4-6%", 0.04, 0.06),
                ("6-8%", 0.06, 0.08),
                ("8-10%", 0.08, 0.10),
                ("10-15%", 0.10, 0.15),
                ("15%+", 0.15, 1.0)
            ]
            
            results = []
            for label, min_edge, max_edge in buckets:
                df = pd.read_sql_query("""
                    SELECT 
                        COUNT(*) as bets,
                        SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                        AVG(CASE WHEN outcome = 'win' THEN 1.0 ELSE 0.0 END) as win_rate,
                        SUM(COALESCE(profit_loss, 0)) as total_profit,
                        SUM(COALESCE(profit_loss, 0)) / NULLIF(SUM(stake_amount), 0) as roi,
                        AVG(stake_amount) as avg_stake
                    FROM trial1306_bets
                    WHERE outcome IS NOT NULL
                    AND edge >= ? AND edge < ?
                """, conn, params=[min_edge, max_edge])
                
                if not df.empty and df['bets'].iloc[0] > 0:
                    results.append({
                        'label': label,
                        'bets': int(df['bets'].iloc[0]),
                        'wins': int(df['wins'].iloc[0]),
                        'losses': int(df['losses'].iloc[0]),
                        'win_rate': df['win_rate'].iloc[0] * 100 if pd.notna(df['win_rate'].iloc[0]) else 0,
                        'profit': df['total_profit'].iloc[0],
                        'roi': df['roi'].iloc[0] * 100 if pd.notna(df['roi'].iloc[0]) else 0,
                        'avg_stake': df['avg_stake'].iloc[0]
                    })
            
            conn.close()
            
            # Populate table
            self.table.setRowCount(len(results))
            for i, result in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(result['label']))
                self.table.setItem(i, 1, QTableWidgetItem(str(result['bets'])))
                self.table.setItem(i, 2, QTableWidgetItem(str(result['wins'])))
                self.table.setItem(i, 3, QTableWidgetItem(str(result['losses'])))
                
                wr_item = QTableWidgetItem(f"{result['win_rate']:.1f}")
                wr_item.setForeground(QColor('#90EE90' if result['win_rate'] > 50 else 'white'))
                self.table.setItem(i, 4, wr_item)
                
                profit_item = QTableWidgetItem(f"${result['profit']:.2f}")
                profit_item.setForeground(QColor('#90EE90' if result['profit'] > 0 else '#FF6B6B'))
                self.table.setItem(i, 5, profit_item)
                
                roi_item = QTableWidgetItem(f"{result['roi']:.1f}")
                roi_item.setForeground(QColor('#90EE90' if result['roi'] > 0 else '#FF6B6B'))
                self.table.setItem(i, 6, roi_item)
                
                self.table.setItem(i, 7, QTableWidgetItem(f"${result['avg_stake']:.2f}"))
            
        except Exception as e:
            print(f"[ERROR] Loading edge analysis: {e}")
            import traceback
            traceback.print_exc()


class CalibrationTab(QWidget):
    """Model calibration and probability accuracy"""
    
    def __init__(self, db_path, parent=None):
        super().__init__(parent)
        self.db_path = db_path
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Calibration")
        refresh_btn.clicked.connect(self.refresh_calibration)
        layout.addWidget(refresh_btn)
        
        # Calibration stats
        self.stats_label = QLabel("No data")
        self.stats_label.setStyleSheet("padding: 10px; background-color: #2c3e50; color: white; font-size: 14px;")
        layout.addWidget(self.stats_label)
        
        # Calibration by probability bucket
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Predicted Prob Range", "Predictions", "Actual Win Rate",
            "Calibration Error", "Brier Score", "Expected Wins"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        self.setLayout(layout)
    
    def refresh_calibration(self):
        """Calculate and display calibration metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all graded predictions with probabilities
            df = pd.read_sql_query("""
                SELECT 
                    model_home_prob,
                    model_away_prob,
                    home_team,
                    away_team,
                    actual_winner
                FROM daily_predictions
                WHERE actual_winner IS NOT NULL
            """, conn)
            
            conn.close()
            
            if df.empty:
                self.stats_label.setText("<b>No graded predictions yet</b>")
                return
            
            # Convert to binary predictions
            predictions = []
            for _, row in df.iterrows():
                # Home team prediction
                predictions.append({
                    'prob': row['model_home_prob'],
                    'actual': 1 if row['actual_winner'] == row['home_team'] else 0
                })
                # Away team prediction
                predictions.append({
                    'prob': row['model_away_prob'],
                    'actual': 1 if row['actual_winner'] == row['away_team'] else 0
                })
            
            pred_df = pd.DataFrame(predictions)
            
            # Overall Brier score
            brier = np.mean((pred_df['prob'] - pred_df['actual']) ** 2)
            
            # Log loss
            epsilon = 1e-15
            pred_df['prob_clipped'] = pred_df['prob'].clip(epsilon, 1 - epsilon)
            log_loss = -np.mean(
                pred_df['actual'] * np.log(pred_df['prob_clipped']) +
                (1 - pred_df['actual']) * np.log(1 - pred_df['prob_clipped'])
            )
            
            self.stats_label.setText(
                f"<b>Total Predictions:</b> {len(pred_df)} | "
                f"<b>Brier Score:</b> {brier:.4f} (lower is better) | "
                f"<b>Log Loss:</b> {log_loss:.4f}"
            )
            
            # Bucket calibration
            buckets = [
                ("0-20%", 0.0, 0.2),
                ("20-40%", 0.2, 0.4),
                ("40-50%", 0.4, 0.5),
                ("50-60%", 0.5, 0.6),
                ("60-80%", 0.6, 0.8),
                ("80-100%", 0.8, 1.0)
            ]
            
            results = []
            for label, min_p, max_p in buckets:
                bucket = pred_df[(pred_df['prob'] >= min_p) & (pred_df['prob'] < max_p)]
                if len(bucket) > 0:
                    avg_pred = bucket['prob'].mean()
                    actual_rate = bucket['actual'].mean()
                    calib_error = abs(avg_pred - actual_rate)
                    bucket_brier = np.mean((bucket['prob'] - bucket['actual']) ** 2)
                    expected_wins = bucket['prob'].sum()
                    actual_wins = bucket['actual'].sum()
                    
                    results.append({
                        'label': label,
                        'count': len(bucket),
                        'actual_rate': actual_rate * 100,
                        'calib_error': calib_error * 100,
                        'brier': bucket_brier,
                        'expected': expected_wins,
                        'actual': actual_wins
                    })
            
            # Populate table
            self.table.setRowCount(len(results))
            for i, result in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(result['label']))
                self.table.setItem(i, 1, QTableWidgetItem(str(result['count'])))
                
                rate_item = QTableWidgetItem(f"{result['actual_rate']:.1f}%")
                self.table.setItem(i, 2, rate_item)
                
                error_item = QTableWidgetItem(f"{result['calib_error']:.1f}%")
                # Color code calibration error
                if result['calib_error'] < 5:
                    error_item.setForeground(QColor('#90EE90'))
                elif result['calib_error'] > 10:
                    error_item.setForeground(QColor('#FF6B6B'))
                self.table.setItem(i, 3, error_item)
                
                self.table.setItem(i, 4, QTableWidgetItem(f"{result['brier']:.4f}"))
                self.table.setItem(i, 5, QTableWidgetItem(f"{result['expected']:.1f} / {result['actual']:.0f}"))
            
        except Exception as e:
            print(f"[ERROR] Loading calibration: {e}")
            import traceback
            traceback.print_exc()

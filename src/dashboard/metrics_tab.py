"""
Metrics Tab for Trial 1306 Dashboard
Displays comprehensive performance metrics, ROI tracking, and bet history
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QHeaderView, QScrollArea, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
import pandas as pd
from datetime import datetime


class MetricsTab(QWidget):
    """Dashboard tab showing Trial 1306 performance metrics"""
    
    def __init__(self, prediction_engine):
        super().__init__()
        self.prediction_engine = prediction_engine
        self.bet_tracker = prediction_engine.bet_tracker if hasattr(prediction_engine, 'bet_tracker') else None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the metrics tab UI"""
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("📊 Trial 1306 Performance Metrics")
        header.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        header.setStyleSheet("padding: 10px; background-color: #34495e; color: white;")
        layout.addWidget(header)
        
        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout()
        
        # Top row: Overall metrics
        metrics_row = QHBoxLayout()
        
        # Overall Performance
        self.overall_box = self._create_metrics_box("📈 Overall Performance")
        metrics_row.addWidget(self.overall_box)
        
        # Financial Metrics
        self.financial_box = self._create_metrics_box("💰 Financial")
        metrics_row.addWidget(self.financial_box)
        
        # Calibration Metrics
        self.calibration_box = self._create_metrics_box("🎯 Calibration")
        metrics_row.addWidget(self.calibration_box)
        
        content_layout.addLayout(metrics_row)
        
        # Bankroll chart (progress bar visual)
        bankroll_group = QGroupBox("💵 Bankroll Tracking")
        bankroll_layout = QVBoxLayout()
        
        self.bankroll_label = QLabel("Current: $2,200.00")
        self.bankroll_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        bankroll_layout.addWidget(self.bankroll_label)
        
        self.bankroll_bar = QProgressBar()
        self.bankroll_bar.setMinimum(0)
        self.bankroll_bar.setMaximum(10000)  # Show up to $10k
        self.bankroll_bar.setValue(2200)
        self.bankroll_bar.setFormat("%v / $10,000")
        bankroll_layout.addWidget(self.bankroll_bar)
        
        self.drawdown_label = QLabel("Max Drawdown: 0.0%")
        bankroll_layout.addWidget(self.drawdown_label)
        
        bankroll_group.setLayout(bankroll_layout)
        content_layout.addWidget(bankroll_group)
        
        # Bet Type Performance
        bet_type_group = QGroupBox("📊 Performance by Bet Type")
        bet_type_layout = QVBoxLayout()
        
        self.bet_type_table = QTableWidget()
        self.bet_type_table.setColumnCount(6)
        self.bet_type_table.setHorizontalHeaderLabels([
            "Bet Type", "Bets", "Win Rate", "Total P/L", "ROI", "Avg Edge"
        ])
        self.bet_type_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bet_type_table.setMaximumHeight(150)
        bet_type_layout.addWidget(self.bet_type_table)
        
        bet_type_group.setLayout(bet_type_layout)
        content_layout.addWidget(bet_type_group)
        
        # Recent Bets
        recent_group = QGroupBox("📋 Recent Bets (Last 20)")
        recent_layout = QVBoxLayout()
        
        self.recent_table = QTableWidget()
        self.recent_table.setColumnCount(9)
        self.recent_table.setHorizontalHeaderLabels([
            "Date", "Game", "Pick", "Odds", "Edge", "Stake", "Outcome", "P/L", "Type"
        ])
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        recent_layout.addWidget(self.recent_table)
        
        recent_group.setLayout(recent_layout)
        content_layout.addWidget(recent_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh Metrics")
        refresh_btn.clicked.connect(self.refresh_metrics)
        refresh_btn.setStyleSheet("padding: 10px; font-weight: bold;")
        button_layout.addWidget(refresh_btn)
        
        grade_btn = QPushButton("✅ Grade Yesterday's Bets")
        grade_btn.clicked.connect(self.grade_bets)
        grade_btn.setStyleSheet("padding: 10px; font-weight: bold; background-color: #27ae60; color: white;")
        button_layout.addWidget(grade_btn)
        
        export_btn = QPushButton("📥 Export to CSV")
        export_btn.clicked.connect(self.export_metrics)
        export_btn.setStyleSheet("padding: 10px; font-weight: bold;")
        button_layout.addWidget(export_btn)
        
        content_layout.addLayout(button_layout)
        
        content.setLayout(content_layout)
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        self.setLayout(layout)
        
        # Initial load
        self.refresh_metrics()
    
    def _create_metrics_box(self, title: str) -> QGroupBox:
        """Create a metrics display box"""
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: bold; padding: 10px; }")
        layout = QVBoxLayout()
        
        # Placeholder labels that will be updated
        layout.addWidget(QLabel("Loading..."))
        
        box.setLayout(layout)
        return box
    
    def refresh_metrics(self):
        """Refresh all metrics from database"""
        if not self.bet_tracker:
            self._show_no_tracker_message()
            return
        
        try:
            # Update performance metrics first
            self.bet_tracker.update_metrics()
            
            # Get current metrics
            metrics = self.bet_tracker.get_metrics()
            
            # Update overall performance box
            self._update_overall_metrics(metrics)
            
            # Update financial metrics box
            self._update_financial_metrics(metrics)
            
            # Update calibration metrics box
            self._update_calibration_metrics(metrics)
            
            # Update bankroll display
            self._update_bankroll_display(metrics)
            
            # Update bet type performance
            self._update_bet_type_table()
            
            # Update recent bets
            self._update_recent_bets()
            
            print("[METRICS] Dashboard refreshed")
            
        except Exception as e:
            print(f"[ERROR] Failed to refresh metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_overall_metrics(self, metrics: dict):
        """Update overall performance metrics"""
        layout = self.overall_box.layout()
        
        # Clear existing widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add metrics
        total_bets = metrics.get('total_bets', 0)
        wins = metrics.get('wins', 0)
        losses = metrics.get('losses', 0)
        pending = metrics.get('pending', 0)
        win_rate = metrics.get('win_rate', 0.0)
        
        layout.addWidget(QLabel(f"<b>Total Bets:</b> {total_bets}"))
        layout.addWidget(QLabel(f"<b>Wins:</b> {wins} | <b>Losses:</b> {losses} | <b>Pending:</b> {pending}"))
        
        # Color-coded win rate
        win_rate_label = QLabel(f"<b>Win Rate:</b> {win_rate:.1%}")
        if win_rate >= 0.55:
            win_rate_label.setStyleSheet("color: green; font-size: 14px;")
        elif win_rate >= 0.50:
            win_rate_label.setStyleSheet("color: orange; font-size: 14px;")
        else:
            win_rate_label.setStyleSheet("color: red; font-size: 14px;")
        layout.addWidget(win_rate_label)
        
        avg_edge = metrics.get('average_edge', 0.0)
        layout.addWidget(QLabel(f"<b>Average Edge:</b> {avg_edge:.2%}"))
    
    def _update_financial_metrics(self, metrics: dict):
        """Update financial metrics"""
        layout = self.financial_box.layout()
        
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        roi = metrics.get('roi', 0.0)
        total_pl = metrics.get('total_profit_loss', 0.0)
        total_staked = metrics.get('total_staked', 0.0)
        avg_stake = metrics.get('average_stake', 0.0)
        
        # Color-coded ROI
        roi_label = QLabel(f"<b>ROI:</b> {roi:.1%}")
        if roi > 0:
            roi_label.setStyleSheet("color: green; font-size: 16px; font-weight: bold;")
        else:
            roi_label.setStyleSheet("color: red; font-size: 16px; font-weight: bold;")
        layout.addWidget(roi_label)
        
        # Profit/Loss
        pl_label = QLabel(f"<b>Total P/L:</b> ${total_pl:+,.2f}")
        if total_pl > 0:
            pl_label.setStyleSheet("color: green;")
        else:
            pl_label.setStyleSheet("color: red;")
        layout.addWidget(pl_label)
        
        layout.addWidget(QLabel(f"<b>Total Staked:</b> ${total_staked:,.2f}"))
        layout.addWidget(QLabel(f"<b>Avg Stake:</b> ${avg_stake:.2f}"))
    
    def _update_calibration_metrics(self, metrics: dict):
        """Update calibration metrics"""
        layout = self.calibration_box.layout()
        
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        brier = metrics.get('brier_score')
        log_loss = metrics.get('log_loss')
        avg_model_prob = metrics.get('average_model_prob', 0.0)
        
        if brier is not None:
            brier_label = QLabel(f"<b>Brier Score:</b> {brier:.4f}")
            if brier < 0.20:
                brier_label.setStyleSheet("color: green;")
            elif brier < 0.25:
                brier_label.setStyleSheet("color: orange;")
            else:
                brier_label.setStyleSheet("color: red;")
            layout.addWidget(brier_label)
        else:
            layout.addWidget(QLabel("<b>Brier Score:</b> N/A"))
        
        if log_loss is not None:
            layout.addWidget(QLabel(f"<b>Log Loss:</b> {log_loss:.4f}"))
        else:
            layout.addWidget(QLabel("<b>Log Loss:</b> N/A"))
        
        layout.addWidget(QLabel(f"<b>Avg Model Prob:</b> {avg_model_prob:.2%}"))
    
    def _update_bankroll_display(self, metrics: dict):
        """Update bankroll progress bar"""
        current = metrics.get('current_bankroll', 2200.0)
        starting = metrics.get('starting_bankroll', 2200.0)
        peak = metrics.get('peak_bankroll', starting)
        max_dd = metrics.get('max_drawdown', 0.0)
        
        self.bankroll_label.setText(f"Current: ${current:,.2f} (Starting: ${starting:,.2f})")
        self.bankroll_bar.setValue(int(current))
        
        # Color based on performance
        if current >= starting:
            self.bankroll_bar.setStyleSheet("QProgressBar::chunk { background-color: #27ae60; }")
        else:
            self.bankroll_bar.setStyleSheet("QProgressBar::chunk { background-color: #e74c3c; }")
        
        dd_label_text = f"Max Drawdown: {max_dd:.1%} | Peak: ${peak:,.2f}"
        if max_dd > 0.20:
            self.drawdown_label.setStyleSheet("color: red; font-weight: bold;")
        elif max_dd > 0.10:
            self.drawdown_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.drawdown_label.setStyleSheet("color: green; font-weight: bold;")
        self.drawdown_label.setText(dd_label_text)
    
    def _update_bet_type_table(self):
        """Update bet type performance table"""
        if not self.bet_tracker:
            return
        
        try:
            df = self.bet_tracker.get_performance_by_type()
            
            self.bet_type_table.setRowCount(len(df))
            
            for i, row in df.iterrows():
                self.bet_type_table.setItem(i, 0, QTableWidgetItem(str(row['bet_type'])))
                self.bet_type_table.setItem(i, 1, QTableWidgetItem(str(row['total_bets'])))
                self.bet_type_table.setItem(i, 2, QTableWidgetItem(f"{row['win_rate']:.1%}"))
                self.bet_type_table.setItem(i, 3, QTableWidgetItem(f"${row['total_profit']:+,.2f}"))
                self.bet_type_table.setItem(i, 4, QTableWidgetItem(f"{row['roi']:.1%}"))
                self.bet_type_table.setItem(i, 5, QTableWidgetItem(f"{row['avg_edge']:.2%}"))
        
        except Exception as e:
            print(f"[ERROR] Failed to update bet type table: {e}")
    
    def _update_recent_bets(self):
        """Update recent bets table"""
        if not self.bet_tracker:
            return
        
        try:
            df = self.bet_tracker.get_recent_bets(limit=20)
            
            self.recent_table.setRowCount(len(df))
            
            for i, row in df.iterrows():
                self.recent_table.setItem(i, 0, QTableWidgetItem(str(row['game_date'])))
                self.recent_table.setItem(i, 1, QTableWidgetItem(f"{row['away_team']} @ {row['home_team']}"))
                self.recent_table.setItem(i, 2, QTableWidgetItem(str(row['predicted_winner'])))
                self.recent_table.setItem(i, 3, QTableWidgetItem(f"{row['market_odds']:+d}"))
                self.recent_table.setItem(i, 4, QTableWidgetItem(f"{row['edge']:.1%}"))
                self.recent_table.setItem(i, 5, QTableWidgetItem(f"${row['stake_amount']:.2f}"))
                
                outcome = str(row['outcome']) if pd.notna(row['outcome']) else 'PENDING'
                outcome_item = QTableWidgetItem(outcome)
                if outcome == 'WIN':
                    outcome_item.setBackground(QColor(39, 174, 96, 80))  # Green
                elif outcome == 'LOSS':
                    outcome_item.setBackground(QColor(231, 76, 60, 80))  # Red
                self.recent_table.setItem(i, 6, outcome_item)
                
                pl = row['profit_loss'] if pd.notna(row['profit_loss']) else 0.0
                pl_item = QTableWidgetItem(f"${pl:+,.2f}" if pl != 0 else "-")
                if pl > 0:
                    pl_item.setForeground(QColor(39, 174, 96))
                elif pl < 0:
                    pl_item.setForeground(QColor(231, 76, 60))
                self.recent_table.setItem(i, 7, pl_item)
                
                self.recent_table.setItem(i, 8, QTableWidgetItem(str(row['bet_type'])))
        
        except Exception as e:
            print(f"[ERROR] Failed to update recent bets: {e}")
    
    def grade_bets(self):
        """Grade bets from yesterday"""
        if not self.bet_tracker:
            print("[ERROR] BetTracker not available")
            return
        
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).date().isoformat()
        
        try:
            graded = self.bet_tracker.grade_bets(yesterday)
            print(f"[GRADING] Graded {graded} bets from {yesterday}")
            
            # Refresh metrics after grading
            self.refresh_metrics()
            
        except Exception as e:
            print(f"[ERROR] Failed to grade bets: {e}")
    
    def export_metrics(self):
        """Export metrics to CSV"""
        if not self.bet_tracker:
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Export recent bets
            df = self.bet_tracker.get_recent_bets(limit=1000)
            filename = f"trial1306_bets_{timestamp}.csv"
            df.to_csv(filename, index=False)
            
            print(f"[EXPORT] Saved to {filename}")
            
        except Exception as e:
            print(f"[ERROR] Failed to export: {e}")
    
    def _show_no_tracker_message(self):
        """Show message when BetTracker is not available"""
        layout = self.layout()
        label = QLabel("⚠️ BetTracker not available. Please ensure src/core/bet_tracker.py is loaded.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: red; font-size: 14px; padding: 50px;")
        layout.addWidget(label)

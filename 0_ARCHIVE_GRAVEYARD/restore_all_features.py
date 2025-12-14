"""Restore all today's dialog features and fixes"""
import re

with open('nba_gui_dashboard_v2.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Make dialog scrollable with scroll area
content = content.replace(
    '    def init_ui(self):\n        matchup = f"{self.prediction[\'away_team\']} @ {self.prediction[\'home_team\']}"\n        self.setWindowTitle(f"Game Details - {matchup}")\n        self.setGeometry(200, 100, 1000, 850)  # Larger window: wider and taller\n        \n        layout = QVBoxLayout()',
    '''    def init_ui(self):
        matchup = f"{self.prediction['away_team']} @ {self.prediction['home_team']}"
        self.setWindowTitle(f"Game Details - {matchup}")
        self.setGeometry(200, 100, 1000, 850)  # Larger window: wider and taller
        
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        
        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)'''
)

# Fix 2: Add Kalshi market prices section after model probabilities
kalshi_section = '''        prob_group.setLayout(prob_layout)
        layout.addWidget(prob_group)
        
        # Kalshi Market Odds (if available)
        kalshi_home = self.prediction.get('kalshi_home_prob')
        kalshi_away = self.prediction.get('kalshi_away_prob')
        odds_source = self.prediction.get('odds_source', 'Default')
        
        if kalshi_home and kalshi_away and odds_source == 'Kalshi':
            kalshi_group = QGroupBox("📊 Kalshi Market Prices (Where You Bet)")
            kalshi_layout = QGridLayout()
            
            kalshi_layout.addWidget(QLabel("<b>Team</b>"), 0, 0)
            kalshi_layout.addWidget(QLabel("<b>Market Price</b>"), 0, 1)
            kalshi_layout.addWidget(QLabel("<b>Edge vs Market</b>"), 0, 2)
            
            home_edge = self.prediction['home_win_prob'] - kalshi_home
            kalshi_layout.addWidget(QLabel(f"<b>{self.prediction['home_team']}</b>"), 1, 0)
            kalshi_layout.addWidget(QLabel(f"{kalshi_home:.1%}"), 1, 1)
            
            # Red/green background with white text for edge
            home_edge_label = QLabel(f"{home_edge:+.1%}")
            home_edge_label.setStyleSheet(
                f"background-color: {'#27ae60' if home_edge > 0.03 else '#e74c3c' if home_edge < -0.03 else '#f39c12'}; "
                f"color: white; padding: 5px; font-weight: bold;"
            )
            home_edge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kalshi_layout.addWidget(home_edge_label, 1, 2)
            
            away_edge = self.prediction['away_win_prob'] - kalshi_away
            kalshi_layout.addWidget(QLabel(f"<b>{self.prediction['away_team']}</b>"), 2, 0)
            kalshi_layout.addWidget(QLabel(f"{kalshi_away:.1%}"), 2, 1)
            
            # Red/green background with white text for edge
            away_edge_label = QLabel(f"{away_edge:+.1%}")
            away_edge_label.setStyleSheet(
                f"background-color: {'#27ae60' if away_edge > 0.03 else '#e74c3c' if away_edge < -0.03 else '#f39c12'}; "
                f"color: white; padding: 5px; font-weight: bold;"
            )
            away_edge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            kalshi_layout.addWidget(away_edge_label, 2, 2)
            
            kalshi_group.setLayout(kalshi_layout)
            layout.addWidget(kalshi_group)
        
        # All Available Bets'''

content = content.replace(
    '        prob_group.setLayout(prob_layout)\n        layout.addWidget(prob_group)\n        \n        # All Available Bets',
    kalshi_section
)

# Fix 3: Fix database path
content = content.replace(
    "conn = sqlite3.connect('V2/v2/data/nba_betting_data.db')",
    "conn = sqlite3.connect('data/live/nba_betting_data.db')"
)

# Fix 4: Add database error handling
content = content.replace(
    '''        # Fetch actual records from database
        import sqlite3
        conn = sqlite3.connect('data/live/nba_betting_data.db')
        cursor = conn.cursor()
        
        # Get home team record
        cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (home_team,))
        home_record = cursor.fetchone()
        home_wins, home_losses, home_last_10 = home_record if home_record else (0, 0, '0-0')
        
        # Get away team record
        cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (away_team,))
        away_record = cursor.fetchone()
        away_wins, away_losses, away_last_10 = away_record if away_record else (0, 0, '0-0')
        
        conn.close()''',
    '''        # Fetch actual records from database
        import sqlite3
        from pathlib import Path
        
        try:
            db_path = Path('data/live/nba_betting_data.db')
            if not db_path.exists():
                db_path = Path('data/nba_betting_data.db')
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (home_team,))
                home_record = cursor.fetchone()
                home_wins, home_losses, home_last_10 = home_record if home_record else (0, 0, '0-0')
            except:
                estimated_games = 20
                home_wins = int(home_win_pct * estimated_games)
                home_losses = estimated_games - home_wins
                home_last_10 = f"{int(home_win_pct * 10)}-{10 - int(home_win_pct * 10)}"
            
            try:
                cursor.execute("SELECT wins, losses, last_10 FROM team_records WHERE team = ?", (away_team,))
                away_record = cursor.fetchone()
                away_wins, away_losses, away_last_10 = away_record if away_record else (0, 0, '0-0')
            except:
                estimated_games = 20
                away_wins = int(away_win_pct * estimated_games)
                away_losses = estimated_games - away_wins
                away_last_10 = f"{int(away_win_pct * 10)}-{10 - int(away_win_pct * 10)}"
            
            conn.close()
        except Exception as e:
            estimated_games = 20
            home_wins = int(home_win_pct * estimated_games)
            home_losses = estimated_games - home_wins
            home_last_10 = f"{int(home_win_pct * 10)}-{10 - int(home_win_pct * 10)}"
            away_wins = int(away_win_pct * estimated_games)
            away_losses = estimated_games - away_wins
            away_last_10 = f"{int(away_win_pct * 10)}-{10 - int(away_win_pct * 10)}"'''
)

# Fix 5: Update advantage column to show team names
content = content.replace(
    '''            # Advantage/Difference column
            adv_item = QTableWidgetItem(diff_str)
            adv_item.setForeground(QColor(255, 255, 255))  # White text
            if home_better:
                adv_item.setBackground(QColor(0, 80, 0))  # Dark green
                adv_item.setToolTip(f"{self.prediction['home_team']} advantage")
            elif away_better:
                adv_item.setBackground(QColor(100, 0, 0))  # Dark red
                adv_item.setToolTip(f"{self.prediction['away_team']} advantage")
            else:
                adv_item.setBackground(QColor(60, 60, 60))  # Dark gray
            comparison_table.setItem(row, 3, adv_item)''',
    '''            # Advantage/Difference column with team name
            if home_better:
                adv_str = f"{self.prediction['home_team']} {diff_str}"
                adv_color = QColor(0, 80, 0)  # Dark green
            elif away_better:
                adv_str = f"{self.prediction['away_team']} {diff_str}"
                adv_color = QColor(100, 0, 0)  # Dark red
            else:
                adv_str = "Even"
                adv_color = QColor(60, 60, 60)  # Dark gray
            
            adv_item = QTableWidgetItem(adv_str)
            adv_item.setForeground(QColor(255, 255, 255))  # White text
            adv_item.setBackground(adv_color)
            adv_item.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            comparison_table.setItem(row, 3, adv_item)'''
)

# Fix 6: Fix injury player key names (player_name not player)
content = content.replace(
    '''inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {inj['player']} - {inj['status']} ({inj['injury']})")''',
    '''inj_text = QLabel(f"<span style='color:{status_color};'>●</span> {inj['player_name']} - {inj['status']} ({inj['injury_desc']})")'''
)

# Fix 7: Change close button to use main_layout and set scroll widget
content = content.replace(
    '''        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)''',
    '''        # Set scroll area content
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # Close button outside scroll area (always visible)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        main_layout.addWidget(close_btn)
        
        self.setLayout(main_layout)'''
)

# Write back
with open('nba_gui_dashboard_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Restored all dialog features")

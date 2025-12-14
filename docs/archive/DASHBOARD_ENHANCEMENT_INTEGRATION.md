# Dashboard Enhancement Integration Guide

## Overview
This guide shows how to integrate the new dashboard enhancements into `NBA_Dashboard_Enhanced_v5.py`.

## New Components Created

### 1. `dashboard_enhancements.py`
- **CalibrationTab**: Reliability curves, Brier score trends, calibration statistics
- **ThemeManager**: Dark/light theme switching with QSS stylesheets
- **TooltipHelper**: Accessibility tooltips with formulas

### 2. `dashboard_metrics_tabs.py`
- **ModelMetricsTab**: Confusion matrices, ROC curves, feature importance
- **RiskGauge**: Custom widget showing risk-of-ruin percentage

### 3. `dashboard_risk_filters.py`
- **RiskManagementTab**: Equity curves, drawdown analysis, risk metrics
- **PredictionFilter**: Interactive filters and CSV export

## Integration Steps

### Step 1: Add Imports to NBA_Dashboard_Enhanced_v5.py

Add to the top of the file (around line 20):

```python
# Dashboard enhancements
try:
    from dashboard_enhancements import CalibrationTab, ThemeManager, TooltipHelper
    from dashboard_metrics_tabs import ModelMetricsTab, RiskGauge
    from dashboard_risk_filters import RiskManagementTab, PredictionFilter
    ENHANCEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"Dashboard enhancements not available: {e}")
    ENHANCEMENTS_AVAILABLE = False
    CalibrationTab = None
    ModelMetricsTab = None
    RiskManagementTab = None
    PredictionFilter = None
    ThemeManager = None
    TooltipHelper = None
```

### Step 2: Add Theme Toggle to __init__

In the `__init__` method of `NBADashboard`, add:

```python
def __init__(self):
    super().__init__()
    
    # Existing initialization...
    self.db_path = "nba_betting_data.db"
    # ... other existing code ...
    
    # Theme management
    self.is_dark_theme = True  # Default to dark
    if ENHANCEMENTS_AVAILABLE and ThemeManager:
        self.setStyleSheet(ThemeManager.get_stylesheet(is_dark=True))
    
    self._init_ui()
```

### Step 3: Add New Tabs to Tab Widget

Find where tabs are created (likely in `_init_ui()` method) and add:

```python
def _init_ui(self):
    """Initialize UI components"""
    self.setWindowTitle("NBA Betting Dashboard - Enhanced v5.1")
    self.setGeometry(100, 100, 1600, 900)
    
    # Create central widget and layout
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    main_layout = QVBoxLayout(central_widget)
    
    # Create toolbar with theme toggle
    toolbar = self._create_toolbar()
    main_layout.addWidget(toolbar)
    
    # Create tab widget
    self.tabs = QTabWidget()
    
    # EXISTING TABS (keep as-is)
    # predictions_tab = ...
    # self.tabs.addTab(predictions_tab, "Predictions")
    # ... other existing tabs ...
    
    # NEW TABS - Add these
    if ENHANCEMENTS_AVAILABLE:
        # Calibration tab
        if CalibrationTab:
            self.calibration_tab = CalibrationTab(self.db_path)
            self.tabs.addTab(self.calibration_tab, "📊 Calibration")
        
        # Model Metrics tab
        if ModelMetricsTab:
            self.metrics_tab = ModelMetricsTab(self.db_path, models_dir="models")
            self.tabs.addTab(self.metrics_tab, "📈 Model Metrics")
        
        # Risk Management tab
        if RiskManagementTab:
            initial_bankroll = 10000.0  # Get from config
            self.risk_tab = RiskManagementTab(self.db_path, initial_bankroll)
            self.tabs.addTab(self.risk_tab, "⚠️ Risk Management")
    
    main_layout.addWidget(self.tabs)
```

### Step 4: Add Theme Toggle Toolbar

Create a new method for the toolbar:

```python
def _create_toolbar(self) -> QWidget:
    """Create toolbar with controls"""
    toolbar = QWidget()
    layout = QHBoxLayout(toolbar)
    layout.setContentsMargins(5, 5, 5, 5)
    
    # Title
    title_label = QLabel("NBA Betting System Dashboard")
    title_font = QFont("Arial", 16, QFont.Weight.Bold)
    title_label.setFont(title_font)
    layout.addWidget(title_label)
    
    layout.addStretch()
    
    # Theme toggle button
    if ENHANCEMENTS_AVAILABLE and ThemeManager:
        self.theme_btn = QPushButton("🌙 Dark Theme")
        self.theme_btn.setCheckable(True)
        self.theme_btn.setChecked(True)  # Default dark
        self.theme_btn.clicked.connect(self._toggle_theme)
        if TooltipHelper:
            self.theme_btn.setToolTip("Toggle between dark and light theme")
        layout.addWidget(self.theme_btn)
    
    # Refresh all button
    refresh_all_btn = QPushButton("🔄 Refresh All")
    refresh_all_btn.clicked.connect(self._refresh_all_tabs)
    if TooltipHelper:
        refresh_all_btn.setToolTip("Refresh all dashboard tabs")
    layout.addWidget(refresh_all_btn)
    
    return toolbar

def _toggle_theme(self):
    """Toggle between dark and light theme"""
    if not ENHANCEMENTS_AVAILABLE or not ThemeManager:
        return
    
    self.is_dark_theme = not self.is_dark_theme
    stylesheet = ThemeManager.get_stylesheet(is_dark=self.is_dark_theme)
    self.setStyleSheet(stylesheet)
    
    # Update button text
    if self.is_dark_theme:
        self.theme_btn.setText("🌙 Dark Theme")
    else:
        self.theme_btn.setText("☀️ Light Theme")

def _refresh_all_tabs(self):
    """Refresh all dashboard tabs"""
    if ENHANCEMENTS_AVAILABLE:
        if hasattr(self, 'calibration_tab'):
            self.calibration_tab.refresh_calibration()
        if hasattr(self, 'metrics_tab'):
            self.metrics_tab.refresh_metrics()
        if hasattr(self, 'risk_tab'):
            self.risk_tab.refresh_risk()
    
    # Refresh existing tabs (add your existing refresh logic here)
    # self.predictions_tab.refresh()
    # etc.
```

### Step 5: Add Interactive Filters to Predictions Tab

In the existing predictions tab, add the filter widget:

```python
def _create_predictions_tab(self) -> QWidget:
    """Create predictions tab with filters"""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    # Add filter widget at top
    if ENHANCEMENTS_AVAILABLE and PredictionFilter:
        self.prediction_filter = PredictionFilter()
        self.prediction_filter.filter_changed.connect(self._apply_prediction_filters)
        self.prediction_filter.export_btn.clicked.connect(self._export_predictions_csv)
        layout.addWidget(self.prediction_filter)
    
    # Existing predictions table
    self.predictions_table = QTableWidget()
    # ... setup table ...
    layout.addWidget(self.predictions_table)
    
    return tab

def _apply_prediction_filters(self):
    """Apply filters to predictions table"""
    if not ENHANCEMENTS_AVAILABLE or not hasattr(self, 'prediction_filter'):
        return
    
    filters = self.prediction_filter.get_filters()
    
    # Hide rows based on filters
    for row in range(self.predictions_table.rowCount()):
        should_hide = False
        
        # Get edge value from table (adjust column index as needed)
        edge_item = self.predictions_table.item(row, 5)  # Assuming edge is column 5
        if edge_item:
            try:
                edge = float(edge_item.text().strip('%'))
                
                # Apply min edge filter
                if edge < filters['min_edge']:
                    should_hide = True
                
                # Apply negative edge filter
                if filters['hide_negative'] and edge < 0:
                    should_hide = True
                
            except ValueError:
                pass
        
        # Get probability if needed
        if filters['hide_low_prob']:
            prob_item = self.predictions_table.item(row, 3)  # Adjust column
            if prob_item:
                try:
                    prob = float(prob_item.text().strip('%'))
                    if prob < 40:
                        should_hide = True
                except ValueError:
                    pass
        
        self.predictions_table.setRowHidden(row, should_hide)

def _export_predictions_csv(self):
    """Export predictions to CSV"""
    try:
        # Get file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Predictions",
            f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        # Query predictions from database
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT 
            p.*,
            g.home_score,
            g.away_score,
            g.total_score
        FROM predictions p
        LEFT JOIN games g ON p.game_id = g.game_id
        WHERE p.game_date >= date('now', '-30 days')
        ORDER BY p.game_date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Export to CSV
        df.to_csv(file_path, index=False)
        
        QMessageBox.information(
            self,
            "Export Successful",
            f"Predictions exported to:\n{file_path}"
        )
        
    except Exception as e:
        QMessageBox.critical(
            self,
            "Export Failed",
            f"Error exporting predictions:\n{str(e)}"
        )
```

### Step 6: Add Tooltips to Existing Widgets

Enhance existing widgets with helpful tooltips:

```python
def _setup_tooltips(self):
    """Add tooltips to widgets for better UX"""
    if not ENHANCEMENTS_AVAILABLE or not TooltipHelper:
        return
    
    # Add tooltips to table headers or buttons
    # Example:
    # self.kelly_column_header.setToolTip(TooltipHelper.get('kelly_stake'))
    # self.edge_column_header.setToolTip(TooltipHelper.get('edge'))
    # self.ev_column_header.setToolTip(TooltipHelper.get('ev'))
```

## Testing Checklist

After integration, test:

- [ ] Dashboard launches without errors
- [ ] All existing tabs still work
- [ ] Calibration tab displays reliability curve and Brier score
- [ ] Model Metrics tab shows confusion matrices and ROC curves
- [ ] Risk Management tab displays equity curve and drawdown
- [ ] Theme toggle switches between dark and light
- [ ] Filters hide/show predictions correctly
- [ ] CSV export works and generates valid file
- [ ] Tooltips display on hover
- [ ] All refresh buttons update their respective tabs

## Fallback Behavior

If matplotlib or other dependencies are missing:
- Tabs will show "Matplotlib not available" message
- Dashboard will still function with existing tabs
- Theme system will gracefully degrade
- Filter widgets will still work for basic filtering

## Performance Considerations

For large datasets (>1000 games):
- Consider pagination in predictions table
- Add loading indicators for slow chart rendering
- Cache feature importance data
- Use background threads for heavy computations

## Customization Options

### Change Theme Colors

Edit `ThemeManager.DARK_THEME` or `LIGHT_THEME` dictionaries in `dashboard_enhancements.py`:

```python
DARK_THEME = {
    'background': '#1e1e1e',  # Main background
    'surface': '#2d2d2d',     # Card/panel background
    'primary': '#0d6efd',     # Accent color
    # ... etc
}
```

### Add New Tooltips

Add to `TooltipHelper.TOOLTIPS` in `dashboard_enhancements.py`:

```python
TOOLTIPS = {
    'your_metric': """
    <b>Your Metric</b><br>
    Explanation here<br>
    <i>Formula: x = y + z</i>
    """,
    # ... etc
}
```

### Adjust Risk Calculations

Modify formulas in `RiskManagementTab._calculate_risk_of_ruin()` in `dashboard_risk_filters.py`.

## Troubleshooting

### Charts not displaying
- Check matplotlib installation: `pip install matplotlib scikit-learn`
- Verify PyQt6 matplotlib backend: `pip install PyQt6`

### Database errors
- Ensure `nba_betting_data.db` exists
- Check table schema matches queries
- Verify predictions table has required columns

### Theme not applying
- Check Qt stylesheet syntax in ThemeManager
- Ensure `setStyleSheet()` called after widget creation

### Filters not working
- Verify column indices in `_apply_prediction_filters()`
- Check data types match filter expectations

## Next Steps

1. **Add Line Movement Visualizer**: Create sparkline charts for each game showing historical odds movement
2. **Implement Auto-sort**: Add automatic sorting by Kelly/EV when data loads
3. **Add More Metrics**: Extend ModelMetricsTab with precision/recall, F1 scores
4. **Mobile-Friendly**: Create responsive layouts for smaller screens
5. **Real-time Updates**: Add auto-refresh timer for live games

## Version History

- **v5.1** - Added Calibration, Model Metrics, Risk tabs + Theme system + Filters
- **v5.0** - Base enhanced dashboard (existing)

## Support

For issues or questions:
1. Check console output for error messages
2. Verify all dependencies installed
3. Test individual tabs in isolation
4. Review database schema compatibility

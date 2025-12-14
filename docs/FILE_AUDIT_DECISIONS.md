# COMPREHENSIVE FILE AUDIT - YOUR DECISIONS REQUIRED

## AUDIT PROCESS
For each file, YOU decide:
- **KEEP** - Production file, actively used
- **ARCHIVE** - Old/deprecated, move to _BROKEN_ARCHIVED
- **DELETE** - Duplicate/test file, can delete
- **NEEDS_REVIEW** - Unclear purpose, need to investigate

---

## ROOT DIRECTORY PYTHON FILES (Excluding V2, archive, examples, Sports_Betting_System)

### DASHBOARDS & UI
1. **nba_gui_dashboard_v2.py** (1461 lines) - Main dashboard using V2 models
   - Status: Currently in use, reverted to V2 models
   - **YOUR DECISION:** ___________

2. **admin_dashboard_v6.py** - ?
   - **YOUR DECISION:** ___________

3. **admin_dashboard.py** - ?
   - **YOUR DECISION:** ___________

4. **dashboard_enhancements.py** - ?
   - **YOUR DECISION:** ___________

5. **dashboard_metrics_tabs.py** - ?
   - **YOUR DECISION:** ___________

6. **dashboard_risk_filters.py** - ?
   - **YOUR DECISION:** ___________

7. **launch_dashboard.py** - ?
   - **YOUR DECISION:** ___________

8. **launch_dashboards.py** - ?
   - **YOUR DECISION:** ___________

9. **NBA_Dashboard_v6_Streamlined.py** - ?
   - **YOUR DECISION:** ___________

### CALIBRATION FILES
10. **calibration_ab_test.py** - ?
    - **YOUR DECISION:** ___________

11. **calibration_scheduler.py** - ?
    - **YOUR DECISION:** ___________

### AUDIT/CLEANUP SCRIPTS (Just Created)
12. **audit_and_cleanup.py** - Just created, archives broken models
    - **YOUR DECISION:** ___________

13. **audit_scripts.py** - ?
    - **YOUR DECISION:** ___________

14. **analyze_cleanup.py** - ?
    - **YOUR DECISION:** ___________

### CHECK/DEBUG SCRIPTS (Diagnostic Tools)
15. **check_all_databases.py**
16. **check_database.py**
17. **check_db_schema.py**
18. **check_db_tables.py**
19. **check_db_records.py**
20. **check_big_db.py**
21. **check_elo.py**
22. **check_elo_distribution.py**
23. **check_elo_fix.py**
24. **check_elo_schema.py**
25. **check_elo_table_structure.py**
26. **check_elo_winner_loser.py**
27. **check_elo_inversion.py**
28. **check_all_elo.py**
29. **check_avg_ratings.py**
30. **check_data_coverage.py**
31. **check_feature_importance.py**
32. **check_feature_order.py**
33. **check_features_abs.py**
34. **check_frozen_injuries.py**
35. **check_game_logs_schema.py**
36. **check_injury_data.py**
37. **check_injury_db.py**
38. **check_injury_feature.py**
39. **check_injury_importance.py**
40. **check_injury_schema.py**
41. **check_market_types.py**
42. **check_missing_team.py**
43. **check_nop_bkn.py**
44. **check_odds.py**
45. **check_okc_uta_history.py**
46. **check_pace_features.py**
47. **check_ewma_efg.py**
48. **check_health.bat** (not .py)

**YOUR DECISION FOR ALL CHECK SCRIPTS:**
- Keep which ones? ___________
- Archive the rest? ___________

### FIX SCRIPTS (Created During Debugging)
49. **fix_duplicates.py**
50. **fix_team_stats_duplicates.py**
51. **fix_game_logs_december.py** - Just created, updated game_logs
52. **fix_elo_polarity.py** - ?

**YOUR DECISION FOR FIX SCRIPTS:** ___________

### DIAGNOSTIC/PROOF SCRIPTS (Just Created)
53. **prove_data_leakage.py** - Proved team_stats has no date filter
54. **walk_forward_backtest.py** - Exposed 77% fake accuracy
55. **walk_forward_backtest_v2.py** - Incomplete V2 test (just created)
56. **shap_gsw_cle_audit.py** - Proved injury impact #69
57. **verify_which_model.py** - Model comparison script
58. **inspect_v2_model.py** - Shows V2 model features

**YOUR DECISION FOR DIAGNOSTIC SCRIPTS:** ___________

### FEATURE FILES
59. **advanced_features.py**
60. **feature_analyzer.py**
61. **feature_diff.py**
62. **feature_extractor_validated.py**
63. **Feature Calculator (with ELO differtial).py** (space in name!)

**YOUR DECISION FOR FEATURE FILES:** ___________

### ATS (Against The Spread) FILES
64. **ats_debug.py**
65. **ats_print_bets.py**
66. **ats_recalc.py**
67. **ats_runner.py**

**YOUR DECISION FOR ATS FILES:** ___________

### ANALYSIS FILES
68. **analyze_predictions.py**
69. **analyze_v1_model.py**
70. **database_audit.py**

**YOUR DECISION FOR ANALYSIS FILES:** ___________

### SYSTEM BUILD FILES
71. **build_professional_system.py**
72. **enhanced_system_demo.py**
73. **finalize_setup.py**
74. **integration_example.py**

**YOUR DECISION FOR BUILD FILES:** ___________

### DATA COLLECTION FILES
75. **fetch_recent_game_data.py**
76. **extend_advanced_stats_2019_2025.py**
77. **nba_team_data.py**

**YOUR DECISION FOR DATA FILES:** ___________

### ELO FILES
78. **elo_service.py**
79. **dynamic elo calculator.py** (space in name!)

**YOUR DECISION FOR ELO FILES:** ___________

### EXECUTION/MIGRATION FILES
80. **execute_cleanup_phase1.py**
81. **execute_migration.py**
82. **create_test_inventory.py**

**YOUR DECISION FOR EXECUTION FILES:** ___________

### KALSHI FILES
83. **FINAL_KALSHI_TEST.py**
84. **kalshi_starter_clients.py**

**YOUR DECISION FOR KALSHI FILES:** ___________

### OTHER STANDALONE FILES
85. **EWMA_PRODUCTION_READY.py**
86. **health_check.py**

**YOUR DECISION FOR STANDALONE FILES:** ___________

---

## INSTRUCTIONS FOR YOU:
1. Review each section above
2. For each file/group, decide: KEEP, ARCHIVE, DELETE, or NEEDS_REVIEW
3. I'll execute your decisions and move to V2 folder next
4. We'll do this systematically until entire workspace is clean

**Fill in your decisions and I'll execute them.**

"""
Monitor Optuna tuning progress
Provides real-time updates on trials, convergence, and best parameters
"""

import optuna
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

PROJECT_ROOT = Path(r"c:\Users\d76do\OneDrive\Documents\New Basketball Model")
MODELS_DIR = PROJECT_ROOT / "models"


def find_latest_study():
    """Find the most recent Optuna study file"""
    study_files = sorted(MODELS_DIR.glob("optuna_study_*.pkl"))
    if not study_files:
        print("❌ No study files found in models/ directory")
        return None
    return study_files[-1]


def monitor_study(study_path, refresh_seconds=30):
    """Monitor study progress with live updates"""
    
    print("=" * 70)
    print("OPTUNA TUNING MONITOR")
    print("=" * 70)
    print(f"Study: {study_path.name}")
    print(f"Refresh: Every {refresh_seconds}s (Ctrl+C to exit)")
    print("=" * 70)
    print()
    
    start_time = datetime.now()
    last_trial_count = 0
    
    try:
        while True:
            # Load study
            try:
                study = optuna.load_study(study_path)
            except Exception as e:
                print(f"⚠️  Could not load study: {e}")
                time.sleep(refresh_seconds)
                continue
            
            # Get stats
            n_trials = len(study.trials)
            if n_trials == 0:
                print("⏳ Waiting for trials to start...")
                time.sleep(refresh_seconds)
                continue
            
            best_trial = study.best_trial
            best_value = study.best_value
            
            # Calculate speed
            elapsed = (datetime.now() - start_time).total_seconds()
            trials_per_hour = (n_trials / elapsed) * 3600 if elapsed > 0 else 0
            
            # Estimate completion
            trials_remaining = 3000 - n_trials
            if trials_per_hour > 0:
                hours_remaining = trials_remaining / trials_per_hour
                eta = datetime.now() + timedelta(hours=hours_remaining)
            else:
                eta = None
            
            # Display update
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ", end="")
            print(f"Trial {n_trials:4d}/3000 | ", end="")
            print(f"Best LogLoss: {best_value:.6f} (Trial #{best_trial.number}) | ", end="")
            print(f"Speed: {trials_per_hour:.1f}/hr | ", end="")
            
            if eta:
                print(f"ETA: {eta.strftime('%H:%M:%S')}", end="")
            
            sys.stdout.flush()
            
            # Show progress update every 10 trials
            if n_trials % 10 == 0 and n_trials != last_trial_count:
                print()  # New line for cleaner output
                print(f"\n📊 Progress Update (Trial {n_trials}):")
                print(f"   Best log loss: {best_value:.6f}")
                print(f"   Best params:")
                for param, value in best_trial.params.items():
                    if isinstance(value, float):
                        print(f"      {param:20s}: {value:.6f}")
                    else:
                        print(f"      {param:20s}: {value}")
                print()
                last_trial_count = n_trials
            
            time.sleep(refresh_seconds)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped by user")
        print(f"\nFinal status:")
        print(f"  Trials completed: {n_trials}/3000")
        print(f"  Best log loss: {best_value:.6f}")
        print(f"  Runtime: {elapsed/3600:.1f} hours")


def show_final_summary(study_path):
    """Show final summary of completed optimization"""
    
    study = optuna.load_study(study_path)
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best log loss: {study.best_value:.6f}")
    
    print(f"\nBest hyperparameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param:20s}: {value:.6f}")
        else:
            print(f"  {param:20s}: {value}")
    
    # Show top 5 trials
    print(f"\nTop 5 Trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"  {i}. Trial #{trial.number:4d}: {trial.value:.6f}")
    
    # Convergence check
    recent_trials = sorted_trials[-100:]  # Last 100 trials
    recent_best = min(t.value for t in recent_trials if t.value is not None)
    overall_best = study.best_value
    
    if abs(recent_best - overall_best) < 0.001:
        print(f"\n✅ Converged: Last 100 trials within 0.001 of best")
    else:
        print(f"\n⚠️  Still improving: Last 100 best={recent_best:.6f} vs overall={overall_best:.6f}")


def main():
    """Main monitoring function"""
    
    # Find study file
    study_path = find_latest_study()
    if study_path is None:
        print("\nNo study file found. Options:")
        print("1. Start tuning: python scripts/optuna_tune_25features.py")
        print("2. Manually specify study path")
        return 1
    
    # Check if tuning is complete
    try:
        study = optuna.load_study(study_path)
        if len(study.trials) >= 3000:
            print(f"✅ Tuning complete ({len(study.trials)} trials)")
            show_final_summary(study_path)
            return 0
    except:
        pass
    
    # Start monitoring
    monitor_study(study_path, refresh_seconds=30)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

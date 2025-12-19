"""Check Optuna optimization progress"""
import optuna
import sys

study_name = "nba_44features_deep_v1"
storage = "sqlite:///models/nba_optuna_44features.db"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"\n{'='*70}")
    print(f"OPTUNA PROGRESS CHECK")
    print(f"{'='*70}\n")
    
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    
    print(f"Study: {study_name}")
    print(f"Total trials: {len(trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Running: {len(running)}")
    
    if completed:
        print(f"\nBest trial so far:")
        best = study.best_trial
        best_auc = best.user_attrs.get('avg_auc', None)
        print(f"  Trial #: {best.number}")
        print(f"  LogLoss: {best.value:.6f}")
        if best_auc:
            print(f"  AUC: {best_auc:.5f}")
        
        print(f"\nLast 5 completed trials:")
        for trial in completed[-5:]:
            auc = trial.user_attrs.get('avg_auc', None)
            auc_str = f"{auc:.5f}" if auc else "N/A"
            print(f"  Trial {trial.number:3d}: LogLoss={trial.value:.6f}, AUC={auc_str}")
    else:
        print("\nNo completed trials yet...")
    
    print(f"\n{'='*70}\n")
    
except Exception as e:
    print(f"Error: {e}")
    print("Optimization may not have started yet or database doesn't exist")
    sys.exit(1)

"""
Model Selector for Da Vinchi Pipeline.

Handles dynamic model selection, performance tracking, and model switching
based on performance criteria and market conditions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks model performance over time"""
    
    def __init__(self, window_days: int = 252):
        self.window_days = window_days
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_days))
        self.last_updated = defaultdict(lambda: None)
    
    def update_metrics(self, model_id: str, metrics: Dict[str, float], timestamp: datetime = None) -> None:
        """Update performance metrics for a model"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store metrics with timestamp
        metric_entry = {
            "timestamp": timestamp,
            "metrics": metrics.copy()
        }
        
        self.metrics_history[model_id].append(metric_entry)
        self.last_updated[model_id] = timestamp
        
        logger.debug(f"Updated metrics for {model_id}: {metrics}")
    
    def get_recent_performance(self, model_id: str, days: int = 30) -> Dict[str, float]:
        """Get recent performance metrics for a model"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            entry for entry in self.metrics_history[model_id]
            if entry["timestamp"] >= cutoff_date
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate average metrics over recent period
        all_metric_names = set()
        for entry in recent_metrics:
            all_metric_names.update(entry["metrics"].keys())
        
        avg_metrics = {}
        for metric_name in all_metric_names:
            values = [entry["metrics"].get(metric_name) for entry in recent_metrics 
                     if entry["metrics"].get(metric_name) is not None]
            if values:
                avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics
    
    def compare_models(self, model_ids: List[str], metric: str = "accuracy") -> Dict[str, float]:
        """Compare recent performance of multiple models"""
        performance = {}
        for model_id in model_ids:
            recent_metrics = self.get_recent_performance(model_id, days=30)
            performance[model_id] = recent_metrics.get(metric, 0.0)
        
        return performance


class ModelSelector:
    """
    Handles dynamic model selection and switching based on performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dynamic_switching = config.get("dynamic_switching", True)
        self.performance_window = config.get("performance_window", 252)  # Days
        self.switching_threshold = config.get("switching_threshold", 0.05)  # 5% performance drop
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker(self.performance_window)
        
        # Model registry
        self.available_models = {}
        self.active_models = {}  # Currently active models per target
        self.model_weights = defaultdict(dict)  # Ensemble weights
        
        # Selection history
        self.selection_history = []
        
        logger.info(f"Model selector initialized (dynamic: {self.dynamic_switching})")
    
    def register_model(self, model_id: str, model_type: str, config: Dict[str, Any]) -> None:
        """Register a model as available for selection"""
        self.available_models[model_id] = {
            "model_type": model_type,
            "config": config,
            "registered_at": datetime.now(),
            "selection_count": 0,
            "last_selected": None
        }
        
        logger.info(f"Registered model {model_id} (type: {model_type})")
    
    def select_best_model(
        self, 
        target: str, 
        available_models: List[str], 
        performance_data: Dict[str, Dict[str, float]] = None
    ) -> str:
        """
        Select the best model for a given target.
        
        Args:
            target: Target variable name
            available_models: List of available model IDs
            performance_data: Optional performance data for models
            
        Returns:
            Selected model ID
        """
        if not available_models:
            raise ValueError("No models available for selection")
        
        # If only one model available, select it
        if len(available_models) == 1:
            selected = available_models[0]
            self._record_selection(target, selected, "only_option")
            return selected
        
        # Use performance data if provided, otherwise use tracked performance
        if performance_data is None:
            performance_data = {
                model_id: self.performance_tracker.get_recent_performance(model_id)
                for model_id in available_models
            }
        
        # Select based on primary metric (accuracy, rmse, etc.)
        primary_metric = self._get_primary_metric(target)
        best_model = self._select_by_performance(available_models, performance_data, primary_metric)
        
        self._record_selection(target, best_model, "performance_based")
        return best_model
    
    def should_switch_model(self, target: str, current_model: str, recent_performance: Dict[str, float]) -> bool:
        """
        Determine if current model should be switched.
        
        Args:
            target: Target variable name
            current_model: Currently active model ID
            recent_performance: Recent performance metrics
            
        Returns:
            True if model should be switched
        """
        if not self.dynamic_switching:
            return False
        
        # Get historical performance
        historical_performance = self.performance_tracker.get_recent_performance(
            current_model, days=60  # Look at longer history
        )
        
        if not historical_performance:
            return False  # No history to compare
        
        primary_metric = self._get_primary_metric(target)
        
        current_score = recent_performance.get(primary_metric, 0.0)
        historical_score = historical_performance.get(primary_metric, 0.0)
        
        # Check if performance has dropped significantly
        if historical_score > 0:
            performance_drop = (historical_score - current_score) / historical_score
            
            if performance_drop > self.switching_threshold:
                logger.warning(f"Model {current_model} performance dropped {performance_drop:.2%} for {target}")
                return True
        
        return False
    
    def create_ensemble(
        self, 
        target: str, 
        model_ids: List[str], 
        performance_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Create ensemble weights based on model performance.
        
        Args:
            target: Target variable name
            model_ids: List of model IDs for ensemble
            performance_data: Performance data for models
            
        Returns:
            Dict mapping model_id to weight
        """
        primary_metric = self._get_primary_metric(target)
        
        # Extract performance scores
        scores = []
        valid_models = []
        
        for model_id in model_ids:
            score = performance_data.get(model_id, {}).get(primary_metric)
            if score is not None:
                scores.append(score)
                valid_models.append(model_id)
        
        if not scores:
            # Equal weights if no performance data
            weight = 1.0 / len(model_ids)
            return {model_id: weight for model_id in model_ids}
        
        # Convert to numpy array
        scores = np.array(scores)
        
        # For accuracy-like metrics (higher is better)
        if primary_metric in ["accuracy", "precision", "recall", "f1"]:
            # Softmax of scores for weights
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            weights = exp_scores / np.sum(exp_scores)
        else:
            # For loss-like metrics (lower is better) - invert scores
            inverted_scores = 1.0 / (scores + 1e-8)  # Avoid division by zero
            weights = inverted_scores / np.sum(inverted_scores)
        
        # Create weight dictionary
        ensemble_weights = dict(zip(valid_models, weights))
        
        # Store ensemble weights
        self.model_weights[target] = ensemble_weights
        
        logger.info(f"Created ensemble for {target}: {ensemble_weights}")
        return ensemble_weights
    
    def get_active_model(self, target: str) -> Optional[str]:
        """Get currently active model for target"""
        return self.active_models.get(target)
    
    def set_active_model(self, target: str, model_id: str) -> None:
        """Set active model for target"""
        self.active_models[target] = model_id
        
        # Update model statistics
        if model_id in self.available_models:
            self.available_models[model_id]["selection_count"] += 1
            self.available_models[model_id]["last_selected"] = datetime.now()
        
        logger.info(f"Set active model for {target}: {model_id}")
    
    def get_model_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about model usage and performance"""
        stats = {}
        
        for model_id, model_info in self.available_models.items():
            recent_perf = self.performance_tracker.get_recent_performance(model_id, days=30)
            
            stats[model_id] = {
                "model_type": model_info["model_type"],
                "selection_count": model_info["selection_count"],
                "last_selected": model_info["last_selected"],
                "recent_performance": recent_perf,
                "is_active": model_id in self.active_models.values()
            }
        
        return stats
    
    def _get_primary_metric(self, target: str) -> str:
        """Determine primary metric for target type"""
        # Heuristic based on target name
        if "cls" in target or "class" in target:
            return "accuracy"
        elif "vol" in target or "volatility" in target:
            return "rmse"
        else:
            return "rmse"  # Default for regression
    
    def _select_by_performance(
        self, 
        model_ids: List[str], 
        performance_data: Dict[str, Dict[str, float]], 
        metric: str
    ) -> str:
        """Select model with best performance for given metric"""
        best_model = None
        best_score = None
        
        for model_id in model_ids:
            score = performance_data.get(model_id, {}).get(metric)
            
            if score is None:
                continue
            
            # For accuracy-like metrics, higher is better
            if metric in ["accuracy", "precision", "recall", "f1"]:
                if best_score is None or score > best_score:
                    best_score = score
                    best_model = model_id
            else:
                # For loss-like metrics, lower is better
                if best_score is None or score < best_score:
                    best_score = score
                    best_model = model_id
        
        # Fallback to first model if no performance data
        if best_model is None:
            best_model = model_ids[0]
            logger.warning(f"No performance data available, selecting first model: {best_model}")
        
        return best_model
    
    def _record_selection(self, target: str, model_id: str, reason: str) -> None:
        """Record model selection for audit trail"""
        selection_record = {
            "timestamp": datetime.now(),
            "target": target,
            "model_id": model_id,
            "reason": reason
        }
        
        self.selection_history.append(selection_record)
        
        # Keep only recent history (last 1000 selections)
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
        
        logger.info(f"Selected {model_id} for {target} (reason: {reason})")
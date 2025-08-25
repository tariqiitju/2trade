#!/usr/bin/env python3
"""
Cleanup Manager for 2Trade System

This script safely removes temporary files, cache, and other disposable data
while protecting important code, documentation, and configuration files.

Author: Consuela Housekeeping Module
Created: 2025-08-24
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple
import shutil
import glob

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class CleanupManager:
    """
    Safe cleanup manager for 2Trade system temporary and cache files.
    
    Protects:
    - All Python files (.py)
    - Documentation directories (docs/, README.md, etc.)
    - Configuration files (.yml, .yaml, .json config files)
    - Source code and important data files
    
    Removes:
    - Temporary files (.tmp, .temp)
    - Cache files and directories
    - Log files (with age limits)
    - Test results and workspace files
    - Backup files (.bak, .backup)
    - OS temporary files
    """
    
    def __init__(self, project_root: str = None, dry_run: bool = False):
        """Initialize the cleanup manager."""
        self.project_root = Path(project_root) if project_root else project_root
        self.dry_run = dry_run
        
        # Setup logging
        self._setup_logging()
        
        # Define patterns for files to delete
        self.temp_patterns = [
            # Temporary files
            "*.tmp", "*.temp", "*~", "*.swp", "*.swo",
            # Cache files
            "__pycache__", "*.pyc", "*.pyo", "*.pyd",
            # Test and workspace files
            "*_workspace", "*workspace*", "test_results_*.json",
            "model_performance_*.json", "accuracy_*.json", "quick_model_results_*.json",
            # Backup files
            "*.bak", "*.backup", "*.old",
            # OS temporary files
            ".DS_Store", "Thumbs.db", "*.lock",
            # Build artifacts
            "*.egg-info", "build/", "dist/",
            # IDE files
            ".vscode/settings.json", ".idea/workspace.xml"
        ]
        
        # Directories that are safe to clean entirely
        self.temp_directories = [
            "__pycache__", ".pytest_cache", ".tox", ".coverage",
            "test_workspace", "ramanujan_workspace/logs", "ramanujan_workspace/results",
            "da_vinchi/workspace", "cache"
        ]
        
        # Protected patterns (never delete)
        self.protected_patterns = [
            # Code files
            "*.py", "*.pyx", "*.pxd",
            # Documentation
            "*.md", "*.rst", "*.txt",
            # Configuration
            "*.yml", "*.yaml", "*.json", "*.ini", "*.cfg", "*.toml",
            # Data files
            "*.parquet", "*.csv", "*.hdf5", "*.h5",
            # Requirements and setup
            "requirements*.txt", "setup.py", "pyproject.toml", "Pipfile*",
            # Git files
            ".git*", ".github/",
            # Environment files
            ".env*", "*.pem", "*.key"
        ]
        
        # Protected directories (never delete contents)
        self.protected_directories = [
            "docs", ".git", ".github", "odins_eye", "ramanujan", "consuela/config",
            "da_vinchi/core", "dave", "venv", "env"
        ]
        
        # Statistics tracking
        self.stats = {
            "files_deleted": 0,
            "directories_deleted": 0,
            "bytes_freed": 0,
            "errors": []
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger(__name__)
        
        if self.dry_run:
            self.logger.info("DRY RUN MODE - No files will actually be deleted")
    
    def is_protected(self, file_path: Path) -> bool:
        """Check if a file or directory is protected from deletion."""
        file_str = str(file_path)
        file_name = file_path.name
        
        # Check if it's in a protected directory
        for protected_dir in self.protected_directories:
            if protected_dir in file_path.parts:
                return True
        
        # Check protected file patterns
        for pattern in self.protected_patterns:
            if file_path.match(pattern) or file_name == pattern:
                return True
        
        # Special protection for important files
        important_files = [
            "CLAUDE.md", "README.md", ".gitignore", "LICENSE", 
            "__init__.py", "requirements.txt"
        ]
        if file_name in important_files:
            return True
        
        return False
    
    def should_delete(self, file_path: Path) -> bool:
        """Check if a file should be deleted based on patterns."""
        if self.is_protected(file_path):
            return False
        
        file_name = file_path.name
        
        # Check if it matches deletion patterns
        for pattern in self.temp_patterns:
            if file_path.match(pattern) or file_name == pattern:
                return True
        
        # Check if it's in a temporary directory
        for temp_dir in self.temp_directories:
            if temp_dir in file_path.parts:
                return True
        
        return False
    
    def get_file_age_days(self, file_path: Path) -> int:
        """Get the age of a file in days."""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return (datetime.now() - mtime).days
        except Exception:
            return 0
    
    def scan_directory(self, directory: Path) -> List[Path]:
        """Scan directory for files that can be safely deleted."""
        files_to_delete = []
        
        if not directory.exists() or not directory.is_dir():
            return files_to_delete
        
        try:
            for item in directory.rglob('*'):
                if item.is_file() and self.should_delete(item):
                    files_to_delete.append(item)
                elif item.is_dir() and item.name in self.temp_directories:
                    # Add all files in temp directories
                    try:
                        for sub_item in item.rglob('*'):
                            if sub_item.is_file() and not self.is_protected(sub_item):
                                files_to_delete.append(sub_item)
                    except Exception as e:
                        self.logger.warning(f"Error scanning temp directory {item}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
            self.stats["errors"].append(f"Scan error: {directory}: {e}")
        
        return files_to_delete
    
    def clean_old_log_files(self, max_age_days: int = 30) -> List[Path]:
        """Find old log files that can be cleaned up."""
        log_files = []
        
        # Common log file locations
        log_patterns = [
            "**/*.log",
            "**/logs/**/*",
            "**/*_workspace/logs/**/*",
            "**/ramanujan_workspace/logs/**/*"
        ]
        
        for pattern in log_patterns:
            try:
                for log_file in self.project_root.glob(pattern):
                    if (log_file.is_file() and 
                        not self.is_protected(log_file) and
                        self.get_file_age_days(log_file) > max_age_days):
                        log_files.append(log_file)
            except Exception as e:
                self.logger.warning(f"Error scanning for logs with pattern {pattern}: {e}")
        
        return log_files
    
    def clean_test_results(self, max_age_days: int = 7) -> List[Path]:
        """Find old test result files."""
        test_files = []
        
        test_patterns = [
            "**/test_results_*.json",
            "**/model_performance_*.json", 
            "**/accuracy_*.json",
            "**/quick_model_results_*.json",
            "**/test_workspace/**/*",
            "**/*_workspace/**/*.json",
            "**/*_workspace/**/*.parquet"
        ]
        
        for pattern in test_patterns:
            try:
                for test_file in self.project_root.glob(pattern):
                    if (test_file.is_file() and 
                        not self.is_protected(test_file) and
                        self.get_file_age_days(test_file) > max_age_days):
                        test_files.append(test_file)
            except Exception as e:
                self.logger.warning(f"Error scanning for test files with pattern {pattern}: {e}")
        
        return test_files
    
    def delete_file(self, file_path: Path) -> bool:
        """Safely delete a single file."""
        try:
            if not file_path.exists():
                return True
            
            file_size = file_path.stat().st_size
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would delete: {file_path} ({file_size} bytes)")
                self.stats["files_deleted"] += 1
                self.stats["bytes_freed"] += file_size
                return True
            
            file_path.unlink()
            self.stats["files_deleted"] += 1
            self.stats["bytes_freed"] += file_size
            self.logger.debug(f"Deleted: {file_path} ({file_size} bytes)")
            return True
            
        except Exception as e:
            error_msg = f"Failed to delete {file_path}: {e}"
            self.logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False
    
    def delete_empty_directories(self, directory: Path) -> None:
        """Remove empty directories recursively."""
        if not directory.exists() or not directory.is_dir():
            return
        
        try:
            for item in directory.rglob('*'):
                if item.is_dir() and not self.is_protected(item):
                    try:
                        if not any(item.iterdir()):  # Directory is empty
                            if self.dry_run:
                                self.logger.info(f"[DRY RUN] Would remove empty directory: {item}")
                                self.stats["directories_deleted"] += 1
                            else:
                                item.rmdir()
                                self.stats["directories_deleted"] += 1
                                self.logger.debug(f"Removed empty directory: {item}")
                    except Exception as e:
                        self.logger.debug(f"Could not remove directory {item}: {e}")
        except Exception as e:
            self.logger.warning(f"Error cleaning empty directories in {directory}: {e}")
    
    def run_cleanup(self, include_logs: bool = False, log_age_days: int = 30, 
                   test_age_days: int = 7) -> Dict:
        """Run the complete cleanup process."""
        self.logger.info("Starting 2Trade system cleanup...")
        start_time = datetime.now()
        
        files_to_delete = []
        
        # 1. Scan for temporary files
        self.logger.info("Scanning for temporary files...")
        files_to_delete.extend(self.scan_directory(self.project_root))
        
        # 2. Clean old test results
        self.logger.info(f"Scanning for test results older than {test_age_days} days...")
        files_to_delete.extend(self.clean_test_results(test_age_days))
        
        # 3. Clean old log files (if enabled)
        if include_logs:
            self.logger.info(f"Scanning for log files older than {log_age_days} days...")
            files_to_delete.extend(self.clean_old_log_files(log_age_days))
        
        # 4. Remove duplicates and sort
        files_to_delete = list(set(files_to_delete))
        files_to_delete.sort()
        
        self.logger.info(f"Found {len(files_to_delete)} files to delete")
        
        # 5. Delete files
        if files_to_delete:
            self.logger.info("Deleting files...")
            for file_path in files_to_delete:
                self.delete_file(file_path)
        
        # 6. Clean empty directories
        self.logger.info("Removing empty directories...")
        self.delete_empty_directories(self.project_root)
        
        # 7. Generate summary
        duration = datetime.now() - start_time
        
        summary = {
            "duration_seconds": duration.total_seconds(),
            "files_deleted": self.stats["files_deleted"],
            "directories_deleted": self.stats["directories_deleted"],
            "bytes_freed": self.stats["bytes_freed"],
            "mb_freed": round(self.stats["bytes_freed"] / (1024 * 1024), 2),
            "errors_count": len(self.stats["errors"]),
            "errors": self.stats["errors"]
        }
        
        self.logger.info("Cleanup completed!")
        self.logger.info(f"Files deleted: {summary['files_deleted']}")
        self.logger.info(f"Directories removed: {summary['directories_deleted']}")
        self.logger.info(f"Space freed: {summary['mb_freed']} MB")
        self.logger.info(f"Duration: {duration.total_seconds():.2f} seconds")
        
        if summary['errors_count'] > 0:
            self.logger.warning(f"Errors encountered: {summary['errors_count']}")
        
        return summary
    
    def preview_cleanup(self) -> Dict:
        """Preview what would be deleted without actually deleting."""
        original_dry_run = self.dry_run
        self.dry_run = True
        
        try:
            summary = self.run_cleanup()
            return summary
        finally:
            self.dry_run = original_dry_run


def main():
    """Main entry point for the cleanup manager script."""
    parser = argparse.ArgumentParser(
        description="Clean up temporary files and cache in 2Trade system"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--include-logs",
        action="store_true", 
        help="Include old log files in cleanup"
    )
    parser.add_argument(
        "--log-age-days",
        type=int,
        default=30,
        help="Delete log files older than N days (default: 30)"
    )
    parser.add_argument(
        "--test-age-days",
        type=int,
        default=7,
        help="Delete test result files older than N days (default: 7)"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        help="Custom project root directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get project root
    root_path = args.project_root if args.project_root else str(project_root)
    
    try:
        # Initialize cleanup manager
        cleanup_manager = CleanupManager(
            project_root=root_path,
            dry_run=args.dry_run
        )
        
        # Run cleanup
        summary = cleanup_manager.run_cleanup(
            include_logs=args.include_logs,
            log_age_days=args.log_age_days,
            test_age_days=args.test_age_days
        )
        
        # Save summary if not dry run
        if not args.dry_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = Path(root_path) / "consuela" / "house-keeping-report" / f"cleanup_summary_{timestamp}.json"
            
            try:
                import json
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
                print(f"\nCleanup summary saved to: {summary_file}")
            except Exception as e:
                print(f"Warning: Could not save cleanup summary: {e}")
        
        # Exit with error code if there were errors
        sys.exit(1 if summary['errors_count'] > 0 else 0)
        
    except Exception as e:
        print(f"Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
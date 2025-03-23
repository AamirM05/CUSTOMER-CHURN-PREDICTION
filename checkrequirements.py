#!/usr/bin/env python
"""
checkrequirements.py - Checks for required packages and installs missing ones.

This script reads requirements from requirements.txt and only installs packages
that are not already installed, preventing conflicts with existing packages.
"""

import subprocess
import sys
import pkg_resources
import logging
from typing import List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/requirements.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_installed_packages() -> Set[Tuple[str, str]]:
    """
    Get a set of installed packages with their versions.
    
    Returns:
        Set of tuples containing (package_name, version)
    """
    installed_packages = {(pkg.key, pkg.version) for pkg in pkg_resources.working_set}
    logger.info(f"Found {len(installed_packages)} packages already installed")
    return installed_packages

def parse_requirements(filename: str = 'requirements.txt') -> List[str]:
    """
    Parse the requirements file.
    
    Args:
        filename: Path to the requirements file
        
    Returns:
        List of requirement strings
    """
    try:
        with open(filename, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Parsed {len(requirements)} requirements from {filename}")
        return requirements
    except FileNotFoundError:
        logger.error(f"Requirements file '{filename}' not found")
        return []

def get_package_name(requirement: str) -> str:
    """
    Extract package name from requirement string.
    
    Args:
        requirement: Requirement string (e.g., 'pandas>=1.3.0')
        
    Returns:
        Package name (e.g., 'pandas')
    """
    # Handle special cases like package[extra]
    if '[' in requirement:
        return requirement.split('[')[0].lower()
    
    # Handle version specifiers
    for operator in ['>=', '==', '<=', '>', '<', '~=', '!=']:
        if operator in requirement:
            return requirement.split(operator)[0].lower()
    
    return requirement.lower()

def check_and_install_requirements(requirements_file: str = 'requirements.txt') -> None:
    """
    Check for missing requirements and install them.
    
    Args:
        requirements_file: Path to the requirements file
    """
    requirements = parse_requirements(requirements_file)
    if not requirements:
        return
    
    installed_packages = get_installed_packages()
    installed_package_names = {name.lower() for name, _ in installed_packages}
    
    # Identify missing packages
    missing_requirements = []
    for req in requirements:
        package_name = get_package_name(req)
        if package_name not in installed_package_names:
            missing_requirements.append(req)
    
    # Install missing packages
    if missing_requirements:
        logger.info(f"Installing {len(missing_requirements)} missing packages: {', '.join(missing_requirements)}")
        for req in missing_requirements:
            try:
                logger.info(f"Installing {req}")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {req}: {e}")
    else:
        logger.info("All required packages are already installed")

if __name__ == "__main__":
    import os
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Check if a specific requirements file was provided
    if len(sys.argv) > 1:
        requirements_file = sys.argv[1]
    else:
        requirements_file = 'requirements.txt'
    
    check_and_install_requirements(requirements_file)

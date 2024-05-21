from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    """Function returns the list of requirements"""

    with open(file_path, "r") as file:
        packages = [line.rstrip("\n").strip() for line in file if line.strip()]
        packages.remove("-e .")
    return packages


setup(
    name="Hamza_Kacem_CPS_use_case_classification",
    version="0.0.1",
    author="Hamza Kacem",
    author_email="hamza.kacem1997@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

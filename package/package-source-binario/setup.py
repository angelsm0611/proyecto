from setuptools import setup, find_packages

setup(
    name="clasificador",
    version="0.1.0",
    description="paquete para el uso de modelos de clasificacion de texto",
    author="Luis angel sanchez marin",
    author_email="la.sanchezm1@uniandes.edu.co",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "clasificador.train": ["data/*.csv"],
    
    },
    entry_points={
        "console_scripts": [
            "entrenar-modelo = clasificador.train.main:main",
        ],
    },
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "peft>=0.5.0",
        "datasets>=2.0.0",
        "accelerate>=0.26.0",
        "tf-keras>=2.15.0",
        "textstat>=0.7.0",
    ],
    python_requires=">=3.8",
)

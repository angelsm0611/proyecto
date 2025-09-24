from setuptools import setup, find_packages

setup(
    name="clasificador",  # Actualizado a 'clasificador'
    version="0.1.0",
    description="A package for biomedical text classification",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "clasificador": ["models/*", "models/biobert_model/*"],  # Ajustado a 'clasificador'
    },
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.30.0",
        "scikit-learn==1.3.0",
        "numpy<2.0.0",
        "datasets>=2.0.0",
        "accelerate>=0.26.0",
        "tf-keras>=2.15.0",  # Agregado
    ],
    python_requires=">=3.8",
    setup_requires=["wheel"],
)
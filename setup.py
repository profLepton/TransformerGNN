from setuptools import setup, find_packages

setup(
    name="TransformerGNN",
    version="0.0.1",
    description="Transformer based faster GNN",
    author="Akshay Kolli",
    author_email="akshaykolli@hotmail.com",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "torch",
        "gensim"
    ],
)
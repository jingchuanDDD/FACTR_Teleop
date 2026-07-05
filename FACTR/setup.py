from setuptools import setup, find_packages

setup(
    name="factr",
    version="0.1",
    packages=find_packages(),
    author="Jason Jingzhou Liu and Yulong Li",
    author_email="liujason@cmu.edu",
    description="FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RaindragonD/factr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
    ],
)

from setuptools import find_packages, setup

setup(
    name="scfields-contact-field",
    py_modules=[
        "dataset",
        "train_contact_field",
        "train_contact_field_lightning",
        "evaluate_contact_field",
        "viz_contact_field",
    ],
    packages=find_packages(),
)

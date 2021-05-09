from setuptools import setup

setup(
    name="bipedal",
    version='0.0.1',
    description="A OpenAI Gym Environment for POPPY humanoid robot.",
    install_requires=['gym', 'pybullet', 'numpy', 'matplotlib', 'torch', 'torchvision']
)

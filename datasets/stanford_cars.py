from torchvision.datasets import StanfordCars

ds = StanfordCars(root='./data', split='train', download=True)
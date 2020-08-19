import torch
from torchvision.models import AlexNet
from models.my_model import *
from torchviz import make_dot
 
x=torch.rand(8,3,256,512)
model=my_model()
y=model(x)


g = make_dot(y)

g.render('espnet_model', view=False) 
import torch
from termcolor import cprint
from .vit import vit
from .deit import deit
from .swin import swin
from .resnet50 import resnet50

def build_model(args):
    
    # using pretrained model
    if args.pretrained == True:
        cprint('WARN => in build_model.py: using pre-trained model: {}'.format(args.arch), "yellow")
    else:
        cprint("WARN => in build_model.py: creating model: {}".format(args.arch), "yellow")
    
    # choosing backbone network
    if args.arch == "resnet50":
         model = resnet50(args.pretrained == True)
    elif args.arch == "vit":
         model = vit(args.pretrained == True)
    elif args.arch == "deit":
        model = deit(args.pretrained == True)
    elif args.arch == "swin":
        model = swin(args.pretrained == True)    
    else:
        raise Exception("Model not defined!!!")
    
    return model



if __name__ == "__main__":

    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='swin')
    parser.add_argument('--pretrained', default=True)
    args = parser.parse_args()

    model = build_model(args)

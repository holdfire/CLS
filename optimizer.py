import torch


def build_optimizer(args):
    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(args.model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(args.model.parameters(), args.lr,
                                    betas=(0.9, 0.999), eps=1e-08, 
                                    weight_decay=args.weight_decay)
    else:
        raise Exception("optimizer type is not defined.")
    
    return optimizer



if __name__ == "__main__":

    # test code
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer-type', default='adamw')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--weight-decay', default=1e-4)
    args = parser.parse_args()

    optimizer = build_optimizer(args)

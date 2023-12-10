from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
import torch.nn.functional as F
from models.mlp import MLP
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import geoopt.manifolds.poincare.math as pmath
import wandb
torch.autograd.set_detect_anomaly(True)


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train(args):
    # wandb.init(project="GNN",config=args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim,_ = data['features'].shape

    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    # model = MLP(input_size, hidden_sizes, output_size)
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        epoch_mae = 0.0
        epoch_loss = 0.0
        for i in data['idx_train']:
            single_features = data['features'][:,:,i]
            single_adj = data['adj_train'][:,:,i]
            embeddings = model.encode(single_features, single_adj)
            # age = model(single_features)
            # train_metrics = model.compute_metrics(embeddings, data, 'train', args)
            train_metrics = model.compute_metrics(embeddings, data, single_adj, single_features, i, args)
            train_metrics['loss'].backward()
            print(train_metrics['mae'])
            # print(data['labels'][i])
            epoch_mae += train_metrics['mae']
            epoch_loss += train_metrics['loss']
            if args.grad_clip is not None:
                max_norm = float(args.grad_clip)
                all_params = list(model.parameters())
                for param in all_params:
                    torch.nn.utils.clip_grad_norm_(param, max_norm)
            optimizer.step()
        lr_scheduler.step()
        epoch_mae = epoch_mae / len(data['idx_train'])
        epoch_loss = epoch_loss / len(data['idx_train'])
        # wandb.log({"epoch": epoch, "mae": epoch_mae, "loss": epoch_loss})


        logging.info(f'Epoch {epoch + 1}:  Average MAE = {epoch_mae}')


        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            epoch_mae_val = 0.0
            epoch_loss_val = 0.0
            for i in data['idx_val']:
                single_features = data['features'][:, :, i]
                single_adj = data['adj_train'][:, :, i]
                embeddings = model.encode(single_features, single_adj)
                val_metrics = model.compute_metrics(embeddings, data, single_adj, single_features, i, args)
                epoch_mae_val += val_metrics['mae']
                epoch_loss_val += val_metrics['loss']

            epoch_mae_val = epoch_mae_val / len(data['idx_train'])
            epoch_loss_val = epoch_loss_val / len(data['idx_train'])
            # wandb.log({"epoch_val": epoch, "mae_val": epoch_mae_val, "loss_val": epoch_loss_val})

            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, single_adj, single_features, i, args)
                if isinstance(embeddings, tuple):
                    best_emb = torch.cat((pmath.logmap0(embeddings[0], c=1.0), embeddings[1]), dim=1).cpu()
                else:
                    best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())

                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        for i in data['idx_test']:
            single_features = data['features'][:, :, i]
            single_adj = data['adj_train'][:, :, i]
            best_emb = model.encode(single_features, single_adj)
            best_test_metrics = model.compute_metrics(embeddings, data, single_adj, single_features, i, args)
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))

    if args.save:
        if isinstance(best_emb, tuple):
            best_emb = torch.cat((pmath.logmap0(best_emb[0], c=1.0), best_emb[1]), dim=1).cpu()
        else:
            best_emb = best_emb.cpu()
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
    print(args)
    # wandb.finish()
    import sys

    sys.exit(0)

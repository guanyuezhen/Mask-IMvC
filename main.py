import random
import torch
import os
import numpy as np
import torch.nn.functional as F
from utils.dataloader import get_data
from models.models import MvAEModel
from utils.logger import BaseLogger
from argparse import ArgumentParser
from models.losses import knn_graph, contrastive_loss
from utils.get_restuls import get_clustering_performance
from utils.clustering_performance import clusteringMetrics


def set_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_name', default="MSRCV1-3view", help='Data directory e.g. MSRCV1')
    parser.add_argument('--train_epochs', type=int, default=3000, help='Max. number of epochs')
    parser.add_argument('--alpha', type=int, default=-3, help='Parameter: alpha')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--log_save_dir', default='./logs/', help='Directory to save the results')
    parser.add_argument('--miss_rate', type=float, default=0.5)
    parser.add_argument('--repeat_times', type=float, default=5)
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    set_seed()
    random_numbers_for_kmeans = random.sample(range(10000), 10)
    print(random_numbers_for_kmeans)
    os.makedirs(args.log_save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    latent_dim = 64
    hid_dims = [192, 128]

    metrics = {
        "ACC": [], "NMI": [], "Purity": [],
        "ARI": [], "Fscore": [], "Precision": [], "Recall": []
    }

    logger = BaseLogger(log_save_dir=args.log_save_dir, log_name=args.data_name + '-' + str(args.miss_rate) + '.csv')
    print("====================== start training ======================")
    print("data_name:", args.data_name, "alpha:", args.alpha)
    logger.write_parameters(args.alpha)
    if args.miss_rate == 0:
        args.repeat_times = 1
    for mask_seed in range(1, args.repeat_times + 1):
        mask, data_x_, data_y, view_num, sample_num, cluster_num, input_dims = get_data(args.data_name, args.miss_rate, mask_seed)
        set_seed()
        data_y = torch.from_numpy(data_y).to(device=device)
        data_x = []
        for v in range(view_num):
            data_x.append(torch.from_numpy(data_x_[v]).to(dtype=torch.float32, device=device))

        observed_mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
        missing_mask = 1 - observed_mask

        graph_x = []
        graph_mask = []
        observed_transition = [] # N x N_o
        missing_transition = [] # N x N_m
        for v in range(view_num):
            observed_sample_num = int(observed_mask[:, v].sum().item())
            if observed_sample_num == 0:
                observed_transition_v = torch.zeros(sample_num, 0).to(dtype=torch.float32, device=device)
            else:
                observed_transition_v = torch.eye(sample_num).to(dtype=torch.float32, device=device)[:, observed_mask[:, v].bool()]

            missing_sample_num = int(missing_mask[:, v].sum().item())
            if missing_sample_num == 0:
                missing_transition_v = torch.zeros(sample_num, 0).to(dtype=torch.float32, device=device)
            else:
                missing_transition_v = torch.eye(sample_num).to(dtype=torch.float32, device=device)[:, missing_mask[:, v].bool()]

            observed_transition.append(observed_transition_v)
            missing_transition.append(missing_transition_v)
            # print(observed_transition[v].size())
            # print(missing_transition[v].size())
            graph_mask.append(observed_mask[:, v:v+1] @ observed_mask[:, v:v+1].t())
            data_x[v] = observed_transition[v].t() @ data_x[v]
            graph_x.append(knn_graph(torch.cdist(data_x[v], data_x[v]) ** 2))

        model = MvAEModel(input_dims,
                          view_num,
                          latent_dim=latent_dim,
                          hid_dims=hid_dims,
                          cluster_num=cluster_num
                          )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        losses = []
        mask_z = 0
        for epoch in range(args.train_epochs):
            # Train
            model.train()
            recs, joint_z, joint_y = model(data_x, graph_x, observed_transition, observed_mask, is_training=True)
            loss_rec = 0

            # graph fusion for contrastive learning
            graph_z = knn_graph(torch.cdist(joint_z, joint_z) ** 2)
            graph_masks = [torch.ones(sample_num, sample_num).to(dtype=torch.float32, device=device)] + graph_mask
            graphs = [graph_z]
            for v in range(view_num):
                graphs.append(observed_transition[v] @ graph_x[v] @ observed_transition[v].t())
            graph_masks = torch.stack(graph_masks, dim=-1)
            graphs = torch.stack(graphs, dim=-1)
            weighted_sum = torch.sum(graph_masks * graphs, dim=-1)
            weights_sum = graph_masks.sum(dim=-1)
            fusion_graph = weighted_sum / weights_sum
            loss_con = contrastive_loss(joint_y, fusion_graph)

            for v in range(view_num):
                loss_rec += F.mse_loss(recs[v], data_x[v])

            optimizer.zero_grad()
            loss = loss_rec + 10 ** args.alpha * loss_con
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                print("dataset: %s, epoch: %d, loss_rec: %.6f, loss_con: %.6f" % (
                    args.data_name, epoch, loss_rec.data.item(), loss_con.data.item()))

        # Test
        model.eval()
        joint_z, joint_y = model(data_x, graph_x, observed_transition, observed_mask)
        mask_seed_metrics = get_clustering_performance(
            joint_z.detach().cpu().numpy(),
            data_y.detach().cpu().numpy(),
            cluster_num,
            random_numbers_for_kmeans
        )
        metrics["ACC"].append(mask_seed_metrics["ACC"])
        metrics["NMI"].append(mask_seed_metrics["NMI"])
        metrics["Purity"].append(mask_seed_metrics["Purity"])
        metrics["ARI"].append(mask_seed_metrics["ARI"])
        metrics["Fscore"].append(mask_seed_metrics["Fscore"])
        metrics["Precision"].append(mask_seed_metrics["Precision"])
        metrics["Recall"].append(mask_seed_metrics["Recall"])

    # Calculate average performance metrics
    average_metrics = {key: np.mean(values) for key, values in metrics.items()}
    std_metrics = {key: np.std(values) for key, values in metrics.items()}
    average_scores = [
        average_metrics["ACC"],
        average_metrics["NMI"],
        average_metrics["Purity"],
        average_metrics["ARI"],
        average_metrics["Fscore"],
        average_metrics["Precision"],
        average_metrics["Recall"],
              ]
    std_scores = [
        std_metrics["ACC"],
        std_metrics["NMI"],
        std_metrics["Purity"],
        std_metrics["ARI"],
        std_metrics["Fscore"],
        std_metrics["Precision"],
        std_metrics["Recall"],
    ]
    logger.write_val(epoch, loss, average_scores)
    logger.write_val(epoch, loss, std_scores)

    logger.close_logger()



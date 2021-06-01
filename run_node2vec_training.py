import argparse
import os

import torch
from tqdm.auto import tqdm, trange
from torch_geometric.nn import Node2Vec
import src.experimentutils as experutils
import src.dataprocessing as dataproc


def main():
    parser = argparse.ArgumentParser(description="Run the training on ethereum graph")
    parser.add_argument("path", type=str, help="path for saving results")
    parser.add_argument("--data_path", type=str, help="path to graph as .gt file",
                        default="data/preprocessed_ethereum_2018_2020.gt")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--batch_size", type=int, help="The batch size", default=128)
    parser.add_argument("--seed", type=int, help="random seed", default=1234)
    parser.add_argument("--use_gpu", action='store_true', help="Use gpu")
    parser.add_argument("--verbosity", type=int, help="verbosity level", default=1)

    args = parser.parse_args()
    os.makedirs(args.path, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    device = torch.device(dev)
    if device.type == 'cuda' and args.verbosity > 0:
        print(f"Using {torch.cuda.get_device_name(0)}")
    experutils.set_seeds(args.seed)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    embedding_dim = 128
    walk_length = 80
    context_size = 10
    walks_per_node = 10
    num_negative_samples = 1
    p_parameter = 1
    q_parameters = [1., 0.5, 2.]

    if args.data_path == "completegraph":
        graph = dataproc.complete_graph(num_nodes=40)
    else:
        graph = dataproc.Graph.read_csv(args.data_path)
    eth_edges = torch.from_numpy(graph.edges)

    for q_idx, q in tqdm(enumerate(q_parameters), total=len(q_parameters)):
        model = Node2Vec(eth_edges, embedding_dim=embedding_dim, walk_length=walk_length,
                         context_size=context_size, walks_per_node=walks_per_node,
                         num_negative_samples=num_negative_samples, p=p_parameter, q=q, sparse=True).to(device)

        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)

        for epoch in trange(num_epochs):
            loss = train()
            print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

            torch.save(model.state_dict(), os.path.join(args.path, f"node2vec_emb_{q_idx}.pth"))


if __name__ == "__main__":
    main()

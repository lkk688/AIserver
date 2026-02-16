        self.dropout = dropout

    def forward(self, x, edge_index):
        # Simple GCN implementation without torch_geometric
        # Normalize adjacency matrix
        adj_norm = self._normalize_adj(edge_index, x.size(0))
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x
    def predict_link(self, z_u, z_v):
        """Predict link probability between two nodes."""
        # Use dot product as similarity measure
        return (z_u * z_v).sum(dim=-1)

    def _normalize_adj(self, edge_index, num_nodes):
        """Normalize adjacency matrix for GCN."""
        # Create sparse adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj[u, v] = 1.0
            adj[v, u] = 1.0  # Undirected

        # Add self-loops
        adj = adj + torch.eye(num_nodes)

        # Degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalized adjacency
        adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        return adj_norm


class GraphSAGELinkPredictor(nn.Module):
    return {'x': x, 'edge_index': edge_index}


    def _normalize_adj(self, edge_index, num_nodes):
        """Normalize adjacency matrix for GCN."""
        # Create sparse adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj[u, v] = 1.0
            adj[v, u] = 1.0  # Undirected

        # Add self-loops
        adj = adj + torch.eye(num_nodes)

        # Degree matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalized adjacency
        adj_norm = deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]
        return adj_norm

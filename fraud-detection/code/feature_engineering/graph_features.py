"""
Graph-Based Features for Fraud Detection
Detects fraud rings and network patterns using graph analysis.
"""
import pandas as pd
import numpy as np
from collections import defaultdict, deque

class GraphFeatureEngineer:
    """Extract features from transaction networks."""
    
    def build_transaction_graph(self, df, user_col='user_id', merchant_col='merchant'):
        """Build graph of user-merchant transactions."""
        graph = defaultdict(set)
        for _, row in df.iterrows():
            graph[row[user_col]].add(row[merchant_col])
        return graph
    
    def connected_components(self, graph):
        """Find connected components using BFS."""
        visited = set()
        components = []
        
        def bfs(start):
            component = []
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                component.append(node)
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return component
        
        for node in graph:
            if node not in visited:
                comp = bfs(node)
                if len(comp) > 1:
                    components.append(comp)
        
        return components
    
    def degree_centrality(self, df, user_col='user_id', merchant_col='merchant'):
        """Calculate network degree for users."""
        df = df.copy()
        degree = df.groupby(user_col)[merchant_col].nunique().reset_index()
        degree.columns = [user_col, 'merchant_diversity']
        return df.merge(degree, on=user_col, how='left')
    
    def clustering_coefficient(self, df, user_col='user_id', merchant_col='merchant'):
        """Calculate transaction network clustering."""
        df = df.copy()
        graph = self.build_transaction_graph(df, user_col, merchant_col)
        
        clustering = {}
        for user, merchants in graph.items():
            k = len(merchants)
            if k < 2:
                clustering[user] = 0
            else:
                edges = 0
                for m1 in merchants:
                    for m2 in merchants:
                        if m1 < m2:
                            edges += 1
                max_edges = k * (k - 1) / 2
                clustering[user] = edges / max_edges if max_edges > 0 else 0
        
        df['clustering_coef'] = df[user_col].map(clustering)
        return df

if __name__ == '__main__':
    print("Graph Features Module")

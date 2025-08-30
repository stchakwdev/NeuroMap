"""
Basic visualization system for concept graphs and neural representations.

This module provides comprehensive visualization tools for concept topology,
neural activations, and model comparisons with both static and interactive plots.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import json


class ConceptVisualizer:
    """
    Main visualization class for concept graphs and neural representations.
    
    This class provides methods for creating static and interactive visualizations
    of concept topologies, activation patterns, and model comparisons.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 8)):
        self.style = style
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 20)
        
        # Set matplotlib style
        plt.style.use('default' if style == 'seaborn' else style)
        sns.set_palette("husl")
    
    def visualize_concept_graph(self,
                              graph: nx.Graph,
                              positions: Dict[int, Tuple[float, float]],
                              node_labels: Optional[Dict[int, str]] = None,
                              true_labels: Optional[torch.Tensor] = None,
                              title: str = "Concept Graph",
                              save_path: Optional[Path] = None,
                              interactive: bool = False) -> Optional[go.Figure]:
        """
        Visualize concept graph with nodes and edges.
        
        Args:
            graph: NetworkX graph
            positions: Node positions dictionary
            node_labels: Optional node labels
            true_labels: Optional ground truth labels for coloring
            title: Plot title
            save_path: Optional path to save figure
            interactive: Whether to create interactive plot
            
        Returns:
            Plotly figure if interactive=True, None otherwise
        """
        if interactive:
            return self._interactive_graph_plot(graph, positions, node_labels, true_labels, title, save_path)
        else:
            return self._static_graph_plot(graph, positions, node_labels, true_labels, title, save_path)
    
    def _static_graph_plot(self,
                          graph: nx.Graph,
                          positions: Dict[int, Tuple[float, float]],
                          node_labels: Optional[Dict[int, str]],
                          true_labels: Optional[torch.Tensor],
                          title: str,
                          save_path: Optional[Path]) -> None:
        """Create static matplotlib graph visualization."""
        plt.figure(figsize=self.figsize)
        
        # Determine node colors
        if true_labels is not None:
            node_colors = [self.color_palette[true_labels[node].item() % len(self.color_palette)] 
                          for node in graph.nodes()]
        else:
            node_colors = self.color_palette[0]
        
        # Draw graph
        nx.draw(graph, positions,
                node_color=node_colors,
                node_size=500,
                edge_color='gray',
                edge_alpha=0.6,
                with_labels=True,
                labels=node_labels if node_labels else {node: str(node) for node in graph.nodes()},
                font_size=10,
                font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _interactive_graph_plot(self,
                               graph: nx.Graph,
                               positions: Dict[int, Tuple[float, float]],
                               node_labels: Optional[Dict[int, str]],
                               true_labels: Optional[torch.Tensor],
                               title: str,
                               save_path: Optional[Path]) -> go.Figure:
        """Create interactive plotly graph visualization."""
        # Prepare node positions
        node_x = [positions[node][0] for node in graph.nodes()]
        node_y = [positions[node][1] for node in graph.nodes()]
        
        # Prepare edge positions
        edge_x = []
        edge_y = []
        for edge in graph.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(125, 125, 125, 0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Determine node colors
        if true_labels is not None:
            node_colors = [true_labels[node].item() for node in graph.nodes()]
            colorscale = 'Viridis'
        else:
            node_colors = ['lightblue'] * len(graph.nodes())
            colorscale = None
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=20,
                color=node_colors,
                colorscale=colorscale,
                line=dict(width=2, color='black')
            ),
            text=[node_labels[node] if node_labels else str(node) for node in graph.nodes()],
            textposition="middle center",
            hovertext=[f"Node {node}<br>Label: {node_labels[node] if node_labels else str(node)}" 
                      for node in graph.nodes()]
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click and drag nodes to explore",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_embeddings_2d(self,
                               embeddings: torch.Tensor,
                               labels: Optional[torch.Tensor] = None,
                               title: str = "2D Embedding Visualization",
                               save_path: Optional[Path] = None,
                               interactive: bool = False) -> Optional[go.Figure]:
        """
        Visualize 2D embeddings with optional labels.
        
        Args:
            embeddings: Tensor of shape (n_points, 2)
            labels: Optional labels for coloring points
            title: Plot title
            save_path: Optional save path
            interactive: Whether to create interactive plot
            
        Returns:
            Plotly figure if interactive=True, None otherwise
        """
        embeddings = embeddings.cpu().numpy()
        
        if interactive:
            return self._interactive_scatter_plot(embeddings, labels, title, save_path)
        else:
            return self._static_scatter_plot(embeddings, labels, title, save_path)
    
    def _static_scatter_plot(self,
                           embeddings: np.ndarray,
                           labels: Optional[torch.Tensor],
                           title: str,
                           save_path: Optional[Path]) -> None:
        """Create static matplotlib scatter plot."""
        plt.figure(figsize=self.figsize)
        
        if labels is not None:
            labels_np = labels.cpu().numpy()
            unique_labels = np.unique(labels_np)
            colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(unique_labels))]
            
            for i, label in enumerate(unique_labels):
                mask = labels_np == label
                plt.scatter(embeddings[mask, 0], embeddings[mask, 1], 
                          c=[colors[i]], label=f'Label {label}', s=100, alpha=0.7)
            plt.legend()
        else:
            plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.7)
        
        # Add point labels
        for i, (x, y) in enumerate(embeddings):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def _interactive_scatter_plot(self,
                                embeddings: np.ndarray,
                                labels: Optional[torch.Tensor],
                                title: str,
                                save_path: Optional[Path]) -> go.Figure:
        """Create interactive plotly scatter plot."""
        if labels is not None:
            labels_np = labels.cpu().numpy()
            color_col = labels_np
        else:
            color_col = ['blue'] * len(embeddings)
        
        fig = go.Figure(data=go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers+text',
            marker=dict(
                size=12,
                color=color_col,
                colorscale='Viridis' if labels is not None else None,
                line=dict(width=2, color='black')
            ),
            text=[str(i) for i in range(len(embeddings))],
            textposition="top center",
            hovertemplate='Point %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            showlegend=False,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_circular_structure(self,
                                   embeddings: torch.Tensor,
                                   p: int,
                                   validation_results: Optional[Dict[str, Any]] = None,
                                   title: str = "Circular Structure Analysis",
                                   save_path: Optional[Path] = None) -> None:
        """
        Visualize circular structure with reference circle.
        
        Args:
            embeddings: 2D embeddings to visualize
            p: Modulus for reference circle
            validation_results: Optional validation metrics
            title: Plot title
            save_path: Optional save path
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Embeddings with reference circle
        embeddings_np = embeddings.cpu().numpy()
        
        # Reference circle
        angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
        circle_x = np.cos(angles)
        circle_y = np.sin(angles)
        
        ax1.plot(circle_x, circle_y, 'k--', alpha=0.5, label='Reference Circle')
        ax1.scatter(circle_x, circle_y, c='red', s=100, alpha=0.7, label='Expected Positions')
        
        # Actual embeddings
        ax1.scatter(embeddings_np[:, 0], embeddings_np[:, 1], 
                   c=range(len(embeddings_np)), cmap='viridis', s=100, alpha=0.8, label='Learned Embeddings')
        
        # Connect corresponding points
        for i in range(min(len(embeddings_np), p)):
            ax1.plot([circle_x[i], embeddings_np[i, 0]], 
                    [circle_y[i], embeddings_np[i, 1]], 'gray', alpha=0.3, linewidth=1)
        
        # Add labels
        for i in range(len(embeddings_np)):
            ax1.annotate(str(i), embeddings_np[i], xytext=(3, 3), textcoords='offset points', fontsize=8)
        for i in range(p):
            ax1.annotate(str(i), (circle_x[i], circle_y[i]), xytext=(-3, -15), textcoords='offset points', fontsize=8, color='red')
        
        ax1.set_title('Learned vs Expected Positions')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot 2: Validation metrics
        if validation_results:
            metrics = []
            values = []
            
            overall_assessment = validation_results.get('overall_assessment', {})
            
            if 'overall_score' in overall_assessment:
                metrics.append('Overall Score')
                values.append(overall_assessment['overall_score'])
            
            if 'circular_ordering' in validation_results:
                circular = validation_results['circular_ordering']
                if 'best_match_score' in circular:
                    metrics.append('Circular Order')
                    values.append(circular['best_match_score'])
            
            if 'distance_consistency' in validation_results:
                distance = validation_results['distance_consistency']
                if 'distance_correlation' in distance:
                    metrics.append('Distance Correlation')
                    values.append(distance['distance_correlation'])
            
            if 'adjacency_structure' in validation_results:
                adjacency = validation_results['adjacency_structure']
                if 'passes_adjacency_test' in adjacency:
                    metrics.append('Adjacency Test')
                    values.append(1.0 if adjacency['passes_adjacency_test'] else 0.0)
            
            if metrics and values:
                bars = ax2.bar(metrics, values, color=self.color_palette[:len(metrics)], alpha=0.7)
                ax2.set_title('Validation Metrics')
                ax2.set_ylabel('Score')
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.3f}', ha='center', va='bottom')
            else:
                ax2.text(0.5, 0.5, 'No validation\nmetrics available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Validation Metrics')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compare_model_representations(self,
                                    representations: Dict[str, torch.Tensor],
                                    labels: Optional[torch.Tensor] = None,
                                    title: str = "Model Comparison",
                                    save_path: Optional[Path] = None,
                                    method: str = 'pca') -> None:
        """
        Compare representations from different models.
        
        Args:
            representations: Dictionary mapping model names to representation tensors
            labels: Optional labels for coloring
            title: Plot title
            save_path: Optional save path
            method: Dimensionality reduction method ('pca', 'tsne')
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        n_models = len(representations)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, reps) in enumerate(representations.items()):
            reps_np = reps.cpu().numpy()
            
            # Reduce to 2D if needed
            if reps_np.shape[1] > 2:
                if method == 'pca':
                    reducer = PCA(n_components=2)
                elif method == 'tsne':
                    reducer = TSNE(n_components=2, random_state=42)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                reps_2d = reducer.fit_transform(reps_np)
            else:
                reps_2d = reps_np
            
            # Plot
            if labels is not None:
                labels_np = labels.cpu().numpy()
                unique_labels = np.unique(labels_np)
                colors = [self.color_palette[j % len(self.color_palette)] for j in range(len(unique_labels))]
                
                for j, label in enumerate(unique_labels):
                    mask = labels_np == label
                    axes[i].scatter(reps_2d[mask, 0], reps_2d[mask, 1], 
                                  c=[colors[j]], label=f'Label {label}', s=80, alpha=0.7)
                
                if i == 0:  # Only show legend for first subplot
                    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                axes[i].scatter(reps_2d[:, 0], reps_2d[:, 1], s=80, alpha=0.7)
            
            # Add point labels
            for j, (x, y) in enumerate(reps_2d):
                axes[i].annotate(str(j), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
            
            axes[i].set_title(f'{model_name}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel(f'{method.upper()} 1')
            axes[i].set_ylabel(f'{method.upper()} 2')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_training_dynamics(self,
                                  histories: Dict[str, Dict[str, List[float]]],
                                  title: str = "Training Dynamics",
                                  save_path: Optional[Path] = None) -> None:
        """
        Visualize training dynamics for multiple models.
        
        Args:
            histories: Dictionary mapping model names to training histories
            title: Plot title
            save_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(histories))]
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = colors[i]
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Training loss
            axes[0, 0].plot(epochs, history['train_loss'], 
                           label=f'{model_name}', color=color, linewidth=2)
            
            # Validation loss
            if 'val_loss' in history:
                axes[0, 1].plot(epochs, history['val_loss'], 
                               label=f'{model_name}', color=color, linewidth=2)
            
            # Training accuracy
            axes[1, 0].plot(epochs, history['train_acc'], 
                           label=f'{model_name}', color=color, linewidth=2)
            
            # Validation accuracy
            if 'val_acc' in history:
                axes[1, 1].plot(epochs, history['val_acc'], 
                               label=f'{model_name}', color=color, linewidth=2)
        
        # Configure subplots
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.05)
        
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1.05)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the visualizer with synthetic data
    visualizer = ConceptVisualizer()
    
    # Create test embeddings (circular arrangement)
    n_points = 8
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    embeddings = torch.tensor([[np.cos(a), np.sin(a)] for a in angles], dtype=torch.float32)
    embeddings += torch.randn_like(embeddings) * 0.1  # Add noise
    
    labels = torch.arange(n_points)
    
    print("Testing visualization system...")
    
    # Test 2D embedding visualization
    print("Creating 2D embedding visualization...")
    visualizer.visualize_embeddings_2d(embeddings, labels, "Test Embeddings")
    
    # Test circular structure visualization
    print("Creating circular structure visualization...")
    fake_validation = {
        'overall_assessment': {'overall_score': 0.85},
        'circular_ordering': {'best_match_score': 0.90},
        'distance_consistency': {'distance_correlation': 0.75},
        'adjacency_structure': {'passes_adjacency_test': True}
    }
    
    visualizer.visualize_circular_structure(embeddings, n_points, fake_validation, "Test Circular Structure")
    
    # Test model comparison
    print("Creating model comparison...")
    representations = {
        'Model A': embeddings,
        'Model B': embeddings + torch.randn_like(embeddings) * 0.2
    }
    visualizer.compare_model_representations(representations, labels, "Model Comparison")
    
    # Test training dynamics
    print("Creating training dynamics visualization...")
    fake_histories = {
        'Model A': {
            'train_loss': [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            'val_loss': [1.1, 0.9, 0.7, 0.5, 0.3, 0.15],
            'train_acc': [0.3, 0.5, 0.7, 0.85, 0.95, 0.99],
            'val_acc': [0.25, 0.45, 0.65, 0.8, 0.9, 0.95]
        },
        'Model B': {
            'train_loss': [1.2, 0.9, 0.7, 0.5, 0.3, 0.2],
            'val_loss': [1.3, 1.0, 0.8, 0.6, 0.4, 0.25],
            'train_acc': [0.2, 0.4, 0.6, 0.75, 0.9, 0.95],
            'val_acc': [0.15, 0.35, 0.55, 0.7, 0.85, 0.9]
        }
    }
    
    visualizer.visualize_training_dynamics(fake_histories, "Training Dynamics")
    
    print("✅ Basic visualization system implementation complete!")
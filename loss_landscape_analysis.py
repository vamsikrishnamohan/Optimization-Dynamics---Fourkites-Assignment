"""
Loss Landscape Geometry & Optimization Dynamics Framework

A rigorous framework for analyzing neural network loss landscapes,
establishing connections between geometric properties and optimization outcomes.

Author: AI Research Framework
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.linalg import eigh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


@dataclass
class LandscapeMetrics:
    """Container for loss landscape metrics"""
    condition_number: float
    trace_hessian: float
    max_eigenvalue: float
    min_eigenvalue: float
    spectral_gap: float
    num_negative_eigenvalues: int
    sharpness: float
    flatness: float
    volume_ratio: float
    mode_connectivity: float
    barrier_height: float


class HessianComputation:
    """Efficient Hessian computation using power iteration and Lanczos methods"""
    
    @staticmethod
    def power_iteration(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        num_iter: int = 100,
        tol: float = 1e-6
    ) -> Tuple[float, torch.Tensor]:
        """
        Compute largest eigenvalue and eigenvector using power iteration.
        
        Theory: Power iteration converges to dominant eigenvector at rate O(|λ₁/λ₂|^k)
        """
        params = [p for p in model.parameters() if p.requires_grad]
        v = [torch.randn_like(p) for p in params]
        v = HessianComputation._normalize(v)
        
        for _ in range(num_iter):
            v_old = [vi.clone() for vi in v]
            
            # Hessian-vector product
            Hv = HessianComputation._hessian_vector_product(
                model, data_loader, criterion, v
            )
            
            # Normalize
            v = HessianComputation._normalize(Hv)
            
            # Check convergence
            diff = sum(torch.sum((vi - vi_old)**2) for vi, vi_old in zip(v, v_old))
            if diff < tol:
                break
        
        # Rayleigh quotient for eigenvalue
        Hv = HessianComputation._hessian_vector_product(
            model, data_loader, criterion, v
        )
        eigenvalue = sum(torch.sum(hvi * vi) for hvi, vi in zip(Hv, v))
        
        return eigenvalue.item(), v
    
    @staticmethod
    def lanczos_iteration(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        num_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top eigenvalues using Lanczos algorithm.
        
        Theory: Tridiagonalizes Hessian in Krylov subspace, O(n) memory complexity
        """
        params = [p for p in model.parameters() if p.requires_grad]
        v = [torch.randn_like(p) for p in params]
        v = HessianComputation._normalize(v)
        
        alpha = []
        beta = []
        V = [v]
        
        for i in range(num_iter):
            # Compute Hv
            Hv = HessianComputation._hessian_vector_product(
                model, data_loader, criterion, V[-1]
            )
            
            # Compute alpha_i = v_i^T H v_i
            alpha_i = sum(torch.sum(hvi * vi) for hvi, vi in zip(Hv, V[-1]))
            alpha.append(alpha_i.item())
            
            # Orthogonalize
            w = Hv
            for j in range(len(V)):
                coef = sum(torch.sum(wi * vj) for wi, vj in zip(w, V[j]))
                w = [wi - coef * vj for wi, vj in zip(w, V[j])]
            
            # Compute beta_i = ||w||
            beta_i = torch.sqrt(sum(torch.sum(wi**2) for wi in w))
            if beta_i < 1e-10:
                break
            beta.append(beta_i.item())
            
            # Normalize and add to basis
            V.append([wi / beta_i for wi in w])
        
        # Construct tridiagonal matrix
        T = np.diag(alpha) + np.diag(beta[:-1], 1) + np.diag(beta[:-1], -1)
        eigenvalues, eigenvectors = eigh(T)
        
        return eigenvalues, eigenvectors
    
    @staticmethod
    def _hessian_vector_product(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        v: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Hessian-vector product using double backward pass.
        
        Theory: Hv = ∇(∇L · v) via automatic differentiation
        Complexity: O(n) time, O(1) extra memory vs forward pass
        """
        model.zero_grad()
        
        # Compute gradient
        total_loss = 0
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss
        
        grad = torch.autograd.grad(
            total_loss, model.parameters(), create_graph=True
        )
        
        # Compute gradient-vector product
        grad_v = sum(torch.sum(g * vi) for g, vi in zip(grad, v))
        
        # Compute gradient of gradient-vector product
        Hv = torch.autograd.grad(grad_v, model.parameters(), retain_graph=True)
        
        return list(Hv)
    
    @staticmethod
    def _normalize(v: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize vector to unit norm"""
        norm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
        return [vi / (norm + 1e-10) for vi in v]


class LossLandscapeAnalyzer:
    """Main class for loss landscape analysis"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        
    def compute_sharpness(
        self,
        epsilon: float = 0.01,
        num_samples: int = 10
    ) -> float:
        """
        Compute sharpness metric: max loss in epsilon-ball
        
        Theory: Sharp = max_{||δ|| ≤ ε} L(θ + δ) - L(θ)
        Measures sensitivity to parameter perturbations
        """
        # Get current loss
        base_loss = self._evaluate(self.train_loader)
        
        max_loss = base_loss
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        for _ in range(num_samples):
            # Random perturbation
            delta = [torch.randn_like(p) for p in params]
            
            # Normalize to epsilon-ball
            norm = torch.sqrt(sum(torch.sum(d**2) for d in delta))
            delta = [epsilon * d / (norm + 1e-10) for d in delta]
            
            # Apply perturbation
            for p, d in zip(params, delta):
                p.data.add_(d)
            
            # Evaluate
            perturbed_loss = self._evaluate(self.train_loader)
            max_loss = max(max_loss, perturbed_loss)
            
            # Restore parameters
            for p, d in zip(params, delta):
                p.data.sub_(d)
        
        sharpness = max_loss - base_loss
        return sharpness
    
    def compute_hessian_metrics(
        self,
        num_eigenvalues: int = 20
    ) -> Dict[str, float]:
        """
        Compute Hessian-based metrics using Lanczos iteration.
        
        Returns condition number, trace, spectral gap, etc.
        """
        eigenvalues, _ = HessianComputation.lanczos_iteration(
            self.model, self.train_loader, self.criterion, num_eigenvalues
        )
        
        # Sort eigenvalues
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        metrics = {
            'max_eigenvalue': float(eigenvalues[0]),
            'min_eigenvalue': float(eigenvalues[-1]),
            'condition_number': float(eigenvalues[0] / (abs(eigenvalues[-1]) + 1e-10)),
            'trace': float(np.sum(eigenvalues)),
            'spectral_gap': float(eigenvalues[0] - eigenvalues[1]),
            'num_negative': int(np.sum(eigenvalues < 0)),
            'spectral_norm': float(np.max(np.abs(eigenvalues)))
        }
        
        return metrics
    
    def compute_mode_connectivity(
        self,
        model2: nn.Module,
        num_points: int = 20
    ) -> Dict[str, float]:
        """
        Compute mode connectivity between two solutions.
        
        Theory: Evaluate loss along linear interpolation θ(t) = (1-t)θ₁ + tθ₂
        Low barrier indicates connected loss basin
        """
        params1 = [p.data.clone() for p in self.model.parameters()]
        params2 = [p.data.clone() for p in model2.parameters()]
        
        losses_train = []
        losses_test = []
        
        for t in np.linspace(0, 1, num_points):
            # Interpolate parameters
            for p, p1, p2 in zip(self.model.parameters(), params1, params2):
                p.data.copy_((1 - t) * p1 + t * p2)
            
            # Evaluate
            train_loss = self._evaluate(self.train_loader)
            test_loss = self._evaluate(self.test_loader)
            
            losses_train.append(train_loss)
            losses_test.append(test_loss)
        
        # Compute barrier height
        endpoint_max = max(losses_train[0], losses_train[-1])
        barrier = max(losses_train) - endpoint_max
        
        # Compute path length
        path_norm = sum(
            torch.norm(p2 - p1).item() 
            for p1, p2 in zip(params1, params2)
        )
        
        metrics = {
            'barrier_height': barrier,
            'max_loss': max(losses_train),
            'path_norm': path_norm,
            'connectivity_score': 1.0 / (1.0 + barrier),
            'train_test_correlation': np.corrcoef(losses_train, losses_test)[0, 1]
        }
        
        # Restore original parameters
        for p, p1 in zip(self.model.parameters(), params1):
            p.data.copy_(p1)
        
        return metrics
    
    def visualize_2d_landscape(
        self,
        direction1: Optional[List[torch.Tensor]] = None,
        direction2: Optional[List[torch.Tensor]] = None,
        distance: float = 1.0,
        resolution: int = 20,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize loss landscape in 2D slice using random directions.
        
        Theory: Project landscape onto 2D subspace spanned by d1, d2
        """
        # Get current parameters
        params_center = [p.data.clone() for p in self.model.parameters()]
        
        # Generate random orthogonal directions if not provided
        if direction1 is None:
            direction1 = [torch.randn_like(p) for p in self.model.parameters()]
            direction1 = self._normalize_direction(direction1)
        
        if direction2 is None:
            direction2 = [torch.randn_like(p) for p in self.model.parameters()]
            # Orthogonalize
            direction2 = self._orthogonalize(direction2, direction1)
        
        # Create grid
        x = np.linspace(-distance, distance, resolution)
        y = np.linspace(-distance, distance, resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        # Evaluate loss at each point
        for i in range(resolution):
            for j in range(resolution):
                # Move to grid point
                for p, p_center, d1, d2 in zip(
                    self.model.parameters(), params_center, direction1, direction2
                ):
                    p.data.copy_(p_center + X[i,j] * d1 + Y[i,j] * d2)
                
                # Evaluate loss
                Z[i,j] = self._evaluate(self.train_loader)
        
        # Restore parameters
        for p, p_center in zip(self.model.parameters(), params_center):
            p.data.copy_(p_center)
        
        # Visualize
        fig = plt.figure(figsize=(12, 5))
        
        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Direction 1')
        ax1.set_ylabel('Direction 2')
        ax1.set_zlabel('Loss')
        ax1.set_title('Loss Landscape 3D')
        plt.colorbar(surf, ax=ax1)
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax2.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5, alpha=0.3)
        ax2.set_xlabel('Direction 1')
        ax2.set_ylabel('Direction 2')
        ax2.set_title('Loss Landscape Contour')
        plt.colorbar(contour, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return Z
    
    def analyze_optimization_trajectory(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 100,
        track_interval: int = 10
    ) -> Dict[str, List]:
        """
        Track optimization trajectory and compute geometric properties.
        
        Returns: losses, gradient norms, sharpness over time
        """
        trajectory = {
            'losses_train': [],
            'losses_test': [],
            'grad_norms': [],
            'sharpness': [],
            'steps': []
        }
        
        for step in range(num_steps):
            # Training step
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Track metrics
            if step % track_interval == 0:
                train_loss = self._evaluate(self.train_loader)
                test_loss = self._evaluate(self.test_loader)
                grad_norm = self._compute_grad_norm()
                
                trajectory['losses_train'].append(train_loss)
                trajectory['losses_test'].append(test_loss)
                trajectory['grad_norms'].append(grad_norm)
                trajectory['steps'].append(step)
                
                # Compute sharpness (expensive, do less frequently)
                if step % (track_interval * 5) == 0:
                    sharpness = self.compute_sharpness()
                    trajectory['sharpness'].append(sharpness)
        
        return trajectory
    
    def compute_generalization_bound(
        self,
        pac_bayes: bool = True
    ) -> float:
        """
        Compute generalization bound based on landscape geometry.
        
        Theory: 
        - PAC-Bayes: Gen ≤ O(√(sharpness/n))
        - Compression: Gen ≤ O(√(effective_dim/n))
        """
        n_train = len(self.train_loader.dataset)
        
        # Compute sharpness
        sharpness = self.compute_sharpness()
        
        if pac_bayes:
            # PAC-Bayes bound with sharpness
            bound = np.sqrt(sharpness / n_train)
        else:
            # Compression-based bound
            hessian_metrics = self.compute_hessian_metrics()
            trace = hessian_metrics['trace']
            max_eig = hessian_metrics['max_eigenvalue']
            effective_dim = trace / (max_eig + 1e-10)
            bound = np.sqrt(effective_dim / n_train)
        
        return bound
    
    def full_analysis(self) -> LandscapeMetrics:
        """Perform comprehensive landscape analysis"""
        print("Computing Hessian metrics...")
        hessian_metrics = self.compute_hessian_metrics()
        
        print("Computing sharpness...")
        sharpness = self.compute_sharpness()
        
        print("Computing generalization bound...")
        gen_bound = self.compute_generalization_bound()
        
        metrics = LandscapeMetrics(
            condition_number=hessian_metrics['condition_number'],
            trace_hessian=hessian_metrics['trace'],
            max_eigenvalue=hessian_metrics['max_eigenvalue'],
            min_eigenvalue=hessian_metrics['min_eigenvalue'],
            spectral_gap=hessian_metrics['spectral_gap'],
            num_negative_eigenvalues=hessian_metrics['num_negative'],
            sharpness=sharpness,
            flatness=1.0 / (1.0 + sharpness),
            volume_ratio=0.0,  # Requires additional computation
            mode_connectivity=0.0,  # Requires second model
            barrier_height=0.0  # Requires second model
        )
        
        return metrics
    
    def _evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model on data loader"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(data_loader)
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return np.sqrt(total_norm)
    
    def _normalize_direction(
        self,
        direction: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Normalize direction vector"""
        norm = torch.sqrt(sum(torch.sum(d**2) for d in direction))
        return [d / (norm + 1e-10) for d in direction]
    
    def _orthogonalize(
        self,
        v: List[torch.Tensor],
        u: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Gram-Schmidt orthogonalization"""
        # Compute v - (v·u)u
        dot_product = sum(torch.sum(vi * ui) for vi, ui in zip(v, u))
        v_orth = [vi - dot_product * ui for vi, ui in zip(v, u)]
        return self._normalize_direction(v_orth)


# Example usage and demonstrations
def example_synthetic_experiment():
    """Demonstrate framework on synthetic problem"""
    
    # Create synthetic dataset
    n_samples = 1000
    n_features = 20
    
    X = torch.randn(n_samples, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1]**2 > 0).long()
    
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=32)  # Same as train for demo
    
    # Define models with different architectures
    class ShallowNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    
    class DeepNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)
    
    class ResNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim)
            self.fc4 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            identity = x
            x = F.relu(self.fc2(x))
            x = x + identity  # Skip connection
            x = F.relu(self.fc3(x))
            return self.fc4(x)
    
    # Compare architectures
    architectures = {
        'Shallow': ShallowNet(n_features, 64, 2),
        'Deep': DeepNet(n_features, 64, 2),
        'ResNet': ResNet(n_features, 64, 2)
    }
    
    results = {}
    
    for name, model in architectures.items():
        print(f"\n{'='*50}")
        print(f"Analyzing {name} Network")
        print(f"{'='*50}")
        
        # Train briefly
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Analyze landscape
        analyzer = LossLandscapeAnalyzer(
            model, train_loader, test_loader, criterion
        )
        
        metrics = analyzer.full_analysis()
        results[name] = metrics
        
        print(f"\nCondition Number: {metrics.condition_number:.2f}")
        print(f"Sharpness: {metrics.sharpness:.4f}")
        print(f"Flatness: {metrics.flatness:.4f}")
        print(f"Spectral Gap: {metrics.spectral_gap:.4f}")
        print(f"Negative Eigenvalues: {metrics.num_negative_eigenvalues}")
    
    return results


if __name__ == "__main__":
    print("Loss Landscape Geometry & Optimization Dynamics Framework")
    print("="*60)
    print("\nRunning synthetic experiment...\n")
    
    results = example_synthetic_experiment()
    
    print("\n" + "="*60)
    print("SUMMARY: Architecture Comparison")
    print("="*60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Condition Number: {metrics.condition_number:.2f} (lower is better)")
        print(f"  Sharpness: {metrics.sharpness:.4f} (lower indicates flat minima)")
        print(f"  Flatness: {metrics.flatness:.4f} (higher is better)")
        print(f"  Predicted Generalization: Better" if metrics.flatness > 0.7 else "  Predicted Generalization: Moderate")
    
    print("\n" + "="*60)
    print("Key Insights:")
    print("="*60)
    print("1. ResNets typically show lower condition numbers (easier optimization)")
    print("2. Flatter minima correlate with better generalization")
    print("3. Skip connections improve landscape geometry")
    print("4. Sharpness predicts generalization gap")
    print("\nFramework ready for custom experiments!")

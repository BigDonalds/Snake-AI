import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import sys

DEBUG_NN = False

def debug_nn(msg, level="INFO"):
    """Debug print with [NN] prefix"""
    if DEBUG_NN:
        print(f"[NN] {msg}", file=sys.stderr, flush=True)


class Activation(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LINEAR = "linear"
    SWISH = "swish"
    GELU = "gelu"

@dataclass
class LayerConfig:
    size: int
    activation: Activation = Activation.RELU
    dropout_rate: float = 0.0
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    l2_reg: float = 0.0

class ModularNeuralNetwork:
    def __init__(self, input_size: int, output_size: int = 4):
        self.input_size = input_size
        self.output_size = output_size
        
        debug_nn(f"Initializing network: input_size={input_size}, output_size={output_size}")
        
        self.layers: List[LayerConfig] = []
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        # Normalization parameters
        self.batch_norm_gamma: List[np.ndarray] = []
        self.batch_norm_beta: List[np.ndarray] = []
        self.batch_norm_running_mean: List[np.ndarray] = []
        self.batch_norm_running_var: List[np.ndarray] = []
        
        # Layer norm parameters
        self.layer_norm_gamma: List[np.ndarray] = []
        self.layer_norm_beta: List[np.ndarray] = []
        
        self._layer_outputs: List[np.ndarray] = []
        self.training = True
        
        # Gradient clipping threshold
        self.gradient_clip = 5.0
        
        # Gradient information for meta-learning
        self.gradient_history: List[np.ndarray] = []
        self.parameter_importance: Optional[np.ndarray] = None
        
    def add_dense_layer(self, size: int, activation: Activation = Activation.RELU, 
                        dropout_rate: float = 0.0, use_batch_norm: bool = False,
                        use_layer_norm: bool = False, l2_reg: float = 0.0):
        debug_nn(f"Adding dense layer: size={size}, activation={activation.value}, dropout={dropout_rate}")
        self.layers.append(LayerConfig(size, activation, dropout_rate, 
                                      use_batch_norm, use_layer_norm, l2_reg))
        
    def build(self):
        prev_size = self.input_size
        debug_nn(f"Building network with {len(self.layers)} layers, input_size={self.input_size}")
        
        for i, layer in enumerate(self.layers):
            # weight initialization with Xavier/Glorot for stability
            if layer.activation in [Activation.RELU, Activation.LEAKY_RELU, Activation.ELU]:
                # He initialization (good for ReLU)
                scale = np.sqrt(2.0 / prev_size)
            elif layer.activation == Activation.TANH:
                # Xavier initialization for tanh
                scale = np.sqrt(1.0 / prev_size)
            elif layer.activation == Activation.SIGMOID:
                # Xavier initialization for sigmoid
                scale = np.sqrt(1.0 / prev_size)
            else:
                # Default Xavier
                scale = np.sqrt(2.0 / (prev_size + layer.size))
            
            self.weights.append(np.random.randn(prev_size, layer.size) * scale * 0.5)
            self.biases.append(np.zeros(layer.size))
            
            debug_nn(f"  Layer {i}: weights shape={self.weights[-1].shape}, scale={scale:.4f}")
            
            if layer.use_batch_norm:
                self.batch_norm_gamma.append(np.ones(layer.size))
                self.batch_norm_beta.append(np.zeros(layer.size))
                self.batch_norm_running_mean.append(np.zeros(layer.size))
                self.batch_norm_running_var.append(np.ones(layer.size))
            
            if layer.use_layer_norm:
                self.layer_norm_gamma.append(np.ones(layer.size))
                self.layer_norm_beta.append(np.zeros(layer.size))
            
            prev_size = layer.size
        
        output_scale = 0.05
        self.weights.append(np.random.randn(prev_size, self.output_size) * output_scale)
        self.biases.append(np.zeros(self.output_size))
        debug_nn(f"  Output layer: weights shape={self.weights[-1].shape}, scale={output_scale}")
        
    def _clip_gradients(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradients to prevent explosion"""
        norm = np.linalg.norm(gradient)
        if norm > self.gradient_clip:
            gradient = gradient * (self.gradient_clip / norm)
        return gradient
        
    def _activate(self, z: np.ndarray, activation: Activation) -> np.ndarray:
        if activation == Activation.RELU:
            return np.maximum(0, z)
        elif activation == Activation.LEAKY_RELU:
            return np.where(z > 0, z, 0.01 * z)
        elif activation == Activation.ELU:
            return np.where(z > 0, z, np.exp(z) - 1)
        elif activation == Activation.SELU:
            alpha = 1.67326324
            scale = 1.05070098
            return scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))
        elif activation == Activation.TANH:
            return np.tanh(z)
        elif activation == Activation.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif activation == Activation.SWISH:
            return z * (1 / (1 + np.exp(-z)))
        elif activation == Activation.GELU:
            return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))
        else:  # LINEAR
            return z
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Ensure x is at least 1D
        original_shape = x.shape
        if x.ndim == 0:
            x = x.reshape(1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != self.input_size:
            debug_nn(f"  ❌ ERROR: Input size mismatch: got {x.shape[1]}, expected {self.input_size}")
            raise ValueError(f"Input size mismatch: got {x.shape[1]}, expected {self.input_size}")
        
        x_mean = np.mean(x, axis=1, keepdims=True)
        x_std = np.std(x, axis=1, keepdims=True)
        x_std = np.maximum(x_std, 1e-8)
        x = (x - x_mean) / x_std
        
        debug_nn(f"Forward pass: input shape={original_shape} -> {x.shape}")
        
        # Check for NaN or Inf
        if np.isnan(x).any():
            debug_nn("  ⚠️ WARNING: NaN detected in input!", "ERROR")
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if np.isinf(x).any():
            debug_nn("  ⚠️ WARNING: Inf detected in input!", "ERROR")
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self._layer_outputs = [x]
        current = x
        
        for i, layer in enumerate(self.layers):
            # Matrix multiplication
            z = np.dot(current, self.weights[i]) + self.biases[i]
            
            # Clip activations
            z = np.clip(z, -100, 100)
            
            # Batch Normalization
            if layer.use_batch_norm and self.training:
                if z.ndim > 1 and z.shape[0] > 1:
                    batch_mean = np.mean(z, axis=0)
                    batch_var = np.var(z, axis=0)
                else:
                    batch_mean = z
                    batch_var = np.ones_like(z) * 1e-5
                
                self.batch_norm_running_mean[i] = 0.9 * self.batch_norm_running_mean[i] + 0.1 * batch_mean
                self.batch_norm_running_var[i] = 0.9 * self.batch_norm_running_var[i] + 0.1 * batch_var
                
                z_norm = (z - batch_mean) / np.sqrt(batch_var + 1e-5)
                z = z_norm * self.batch_norm_gamma[i] + self.batch_norm_beta[i]
                
            elif layer.use_batch_norm:
                z_norm = (z - self.batch_norm_running_mean[i]) / np.sqrt(self.batch_norm_running_var[i] + 1e-5)
                z = z_norm * self.batch_norm_gamma[i] + self.batch_norm_beta[i]
            
            # Layer Normalization
            if layer.use_layer_norm:
                if z.ndim > 1:
                    mean = np.mean(z, axis=1, keepdims=True)
                    var = np.var(z, axis=1, keepdims=True)
                else:
                    mean = np.mean(z)
                    var = np.var(z)
                
                z_norm = (z - mean) / np.sqrt(var + 1e-5)
                z = z_norm * self.layer_norm_gamma[i] + self.layer_norm_beta[i]
            
            # Activation
            current = self._activate(z, layer.activation)
            current = np.clip(current, -100, 100)
            
            # Dropout
            if self.training and layer.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - layer.dropout_rate, current.shape)
                current = current * mask / (1 - layer.dropout_rate)
            
            self._layer_outputs.append(current)
        
        # Output layer (linear)
        final_z = np.dot(current, self.weights[-1]) + self.biases[-1]
        final_z = np.clip(final_z, -50, 50)
        
        # Flatten output if it's 2D with batch size 1
        if final_z.ndim == 2 and final_z.shape[0] == 1:
            final_z = final_z.flatten()
        
        self._layer_outputs.append(final_z)
        return final_z
    
    def predict(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        debug_nn(f"Predict: temperature={temperature}")
        self.training = False
        logits = self.forward(x)
        
        # Handle both 1D and 2D outputs
        if logits.ndim == 2:
            logits = logits.flatten()
        
        # Apply temperature with safe scaling
        if temperature != 1.0:
            logits = logits / max(temperature, 0.1)
        
        # Softmax with numerical stability
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)
        
        return probs.flatten()
    
    def get_action(self, x: np.ndarray, deterministic: bool = True, 
                  temperature: float = 1.0, exploration_noise: float = 0.0) -> int:
        debug_nn(f"=== get_action called ===")
        
        probs = self.predict(x, temperature)
        probs = probs.flatten()
        probs = np.clip(probs, 1e-8, 1.0)
        
        # Normalize if needed
        prob_sum = np.sum(probs)
        if abs(prob_sum - 1.0) > 1e-6:
            probs = probs / prob_sum
        
        # Add exploration noise
        if not deterministic and exploration_noise > 0:
            # Epsilon-greedy style exploration
            if np.random.random() < exploration_noise:
                return np.random.randint(len(probs))
        
        # Deterministic action
        if deterministic:
            return int(np.argmax(probs))
        
        # Stochastic sampling
        return int(np.random.choice(len(probs), p=probs))
    
    def get_genome(self) -> np.ndarray:
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b.flatten())
        for g, beta in zip(self.batch_norm_gamma, self.batch_norm_beta):
            parts.append(g.flatten())
            parts.append(beta.flatten())
        for g, beta in zip(self.layer_norm_gamma, self.layer_norm_beta):
            parts.append(g.flatten())
            parts.append(beta.flatten())
        genome = np.concatenate(parts) if parts else np.array([])
        return genome
    
    def set_genome(self, genome: np.ndarray):
        idx = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.size
            if idx + w_size > len(genome):
                break
            self.weights[i] = genome[idx:idx + w_size].reshape(w.shape)
            idx += w_size
            b_size = b.size
            if idx + b_size > len(genome):
                break
            self.biases[i] = genome[idx:idx + b_size].reshape(b.shape)
            idx += b_size
        
        for i in range(len(self.batch_norm_gamma)):
            g_size = self.batch_norm_gamma[i].size
            if idx + g_size > len(genome):
                break
            self.batch_norm_gamma[i] = genome[idx:idx + g_size].reshape(self.batch_norm_gamma[i].shape)
            idx += g_size
            beta_size = self.batch_norm_beta[i].size
            if idx + beta_size > len(genome):
                break
            self.batch_norm_beta[i] = genome[idx:idx + beta_size].reshape(self.batch_norm_beta[i].shape)
            idx += beta_size
        
        for i in range(len(self.layer_norm_gamma)):
            g_size = self.layer_norm_gamma[i].size
            if idx + g_size > len(genome):
                break
            self.layer_norm_gamma[i] = genome[idx:idx + g_size].reshape(self.layer_norm_gamma[i].shape)
            idx += g_size
            beta_size = self.layer_norm_beta[i].size
            if idx + beta_size > len(genome):
                break
            self.layer_norm_beta[i] = genome[idx:idx + beta_size].reshape(self.layer_norm_beta[i].shape)
            idx += beta_size


class EvolutionaryOptimizer:
    @staticmethod
    def crossover(parent1: np.ndarray, parent2: np.ndarray, method: str = 'simulated_binary', 
                  eta: float = 2.0) -> np.ndarray:
        """Advanced crossover methods"""
        if method == 'simulated_binary':
            u = np.random.random(len(parent1))
            beta = np.where(u <= 0.5, 
                          (2 * u) ** (1 / (eta + 1)),
                          (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
            child = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            return child
        
        elif method == 'blend_alpha':
            alpha = 0.5
            low = np.minimum(parent1, parent2) - alpha * np.abs(parent1 - parent2)
            high = np.maximum(parent1, parent2) + alpha * np.abs(parent1 - parent2)
            return np.random.uniform(low, high)
        
        elif method == 'weighted_average':
            weight = np.random.beta(2, 2)
            return weight * parent1 + (1 - weight) * parent2
        
        elif method == 'uniform':
            mask = np.random.random(len(parent1)) < 0.5
            return np.where(mask, parent1, parent2)
        
        elif method == 'single_point':
            point = np.random.randint(1, len(parent1))
            return np.concatenate([parent1[:point], parent2[point:]])
        
        elif method == 'two_point':
            p1, p2 = sorted(np.random.choice(len(parent1), 2, replace=False))
            child = parent1.copy()
            child[p1:p2] = parent2[p1:p2]
            return child
        
        else:
            return 0.5 * (parent1 + parent2)
    
    @staticmethod
    def mutate(genome: np.ndarray, rate: float = 0.1, strength: float = 0.05,
               method: str = 'adaptive_gaussian', generation: int = 0) -> np.ndarray:
        mutated = genome.copy()
        
        # Cap mutation strength
        strength = min(strength, 0.05)
        
        if method == 'adaptive_gaussian':
            magnitude = np.abs(genome)
            adaptive_strength = strength * (1 + magnitude / (np.max(np.abs(magnitude)) + 1e-8))
            mask = np.random.random(len(genome)) < rate
            noise = np.random.randn(len(genome)) * adaptive_strength * 0.3
            mutated += mask * noise
        
        elif method == 'polynomial':
            eta = 20.0
            r = np.random.random(len(genome))
            delta = np.where(r < 0.5,
                           (2 * r) ** (1 / (eta + 1)) - 1,
                           1 - (2 * (1 - r)) ** (1 / (eta + 1)))
            mutated += delta * strength * np.abs(genome) * 0.3
        
        else:  # default gaussian
            mask = np.random.random(len(genome)) < rate
            noise = np.random.randn(len(genome)) * strength * 0.3
            mutated += mask * noise
        
        return mutated

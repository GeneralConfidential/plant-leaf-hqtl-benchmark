"""Hybrid quantum-classical and classical baseline model builders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
import pennylane as qml
from pennylane import numpy as qnp


def _build_quantum_layers(n_qubits: int, q_depth: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    def h_layer(nqubits: int) -> None:
        for idx in range(nqubits):
            qml.Hadamard(wires=idx)

    def ry_layer(weights) -> None:
        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)

    def entangling_layer(nqubits: int) -> None:
        for i in range(0, nqubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, nqubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev, interface="torch")
    def quantum_net(q_input_features, q_weights_flat):
        q_weights = q_weights_flat.reshape(q_depth, n_qubits)
        h_layer(n_qubits)
        ry_layer(q_input_features)
        for k in range(q_depth):
            entangling_layer(n_qubits)
            ry_layer(q_weights[k])
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

    return quantum_net


class DressedQuantumNet(nn.Module):
    def __init__(
        self,
        n_qubits: int,
        n_classes: int,
        q_depth: int,
        q_delta: float = 0.01,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.quantum_net = _build_quantum_layers(n_qubits, q_depth)
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, n_classes)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * qnp.pi / 2.0

        q_out_list = []
        for elem in q_in:
            # PennyLane default.qubit simulates on CPU; keep autograd via torch interface.
            out = torch.stack(
                self.quantum_net(elem.cpu(), self.q_params.cpu())
            ).float().to(input_features.device)
            q_out_list.append(out)
        q_out = torch.stack(q_out_list)
        return self.post_net(q_out)


def build_hybrid_model(
    n_classes: int,
    device: torch.device,
    *,
    n_qubits: int = 10,
    q_depth: int = 4,
) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = DressedQuantumNet(n_qubits, n_classes, q_depth)
    return model.to(device)


def build_linear_head_model(n_classes: int, device: torch.device) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, n_classes)
    return model.to(device)


def build_resnet18_finetune(n_classes: int, device: torch.device) -> nn.Module:
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def build_densenet121_finetune(n_classes: int, device: torch.device) -> nn.Module:
    weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    model = torchvision.models.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    return model.to(device)


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

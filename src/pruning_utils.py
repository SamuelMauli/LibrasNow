import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tqdm import tqdm


def calculate_saliency(model, data_loader, criterion, device):
    """Calcula a saliência de cada peso como |peso * gradiente|."""
    model.eval()

    # Zera os gradientes
    for param in model.parameters():
        param.grad = torch.zeros_like(param)

    # Acumula gradientes sobre um lote de dados de calibração
    for sequences, labels in tqdm(data_loader, desc="Calculating Saliency"):
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()  # Acumula gradientes
        break  # Um único lote é geralmente suficiente

    saliencies = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module.weight, "grad") and module.weight.grad is not None:
                saliency = (
                    (module.weight.grad.abs() * module.weight.abs())
                    .detach()
                    .cpu()
                    .numpy()
                )
                saliencies.append((name, module, saliency))

    model.zero_grad()
    return saliencies


def saliency_based_pruning(model, amount, data_loader, criterion, device):
    """Aplica poda não estruturada global baseada na saliência."""
    print("\n--- Starting Saliency-Based Pruning ---")
    saliencies = calculate_saliency(model, data_loader, criterion, device)

    parameters_to_prune = []
    for name, module, _ in saliencies:
        parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,  # Usamos L1Unstructured como base, mas a lógica de seleção será a nossa
        amount=amount,
    )

    # Torna a poda permanente
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    print(f"Applied {amount*100:.2f}% global unstructured pruning based on saliency.")
    return model

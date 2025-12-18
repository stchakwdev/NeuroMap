#!/usr/bin/env python3
"""
Complete NeuroMap Analysis Pipeline.

Runs the full mechanistic interpretability analysis:
1. Train or load a model
2. Extract activations and embeddings
3. Analyze Fourier structure
4. Run causal interventions
5. Discover circuits
6. Generate visualizations and reports

Usage:
    python experiments/run_full_analysis.py --modulus 17 --output results/
    python experiments/run_full_analysis.py --model-path models/successful/DirectLookup_Adam_p17.pt
"""

import argparse
import sys
from pathlib import Path
import json
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Dataset.dataset import ModularArithmeticDataset
from models.hooked_transformer import create_hooked_model, NeuroMapTransformerConfig
from models.model_configs import get_config, get_training_config
from analysis.activation_extractor import ActivationExtractor
from analysis.fourier_analysis import FourierAnalyzer, generate_fourier_report
from analysis.causal_intervention import ActivationPatcher, CausalAnalyzer, create_corrupted_inputs
from analysis.faithfulness import FaithfulnessEvaluator
from analysis.circuit_discovery import CircuitDiscoverer, generate_circuit_report
from analysis.concept_extractors import ClusteringExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Run full NeuroMap analysis')
    parser.add_argument('--modulus', '-p', type=int, default=17,
                       help='Modulus for modular arithmetic (default: 17)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pretrained model (optional)')
    parser.add_argument('--output', '-o', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--train-epochs', type=int, default=1000,
                       help='Training epochs if training from scratch')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, use random model for demo')
    return parser.parse_args()


def train_model(model, dataset, epochs, device):
    """Train model on modular arithmetic."""
    print(f"\nTraining model for {epochs} epochs...")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    inputs = dataset.data['inputs'].to(device)
    targets = dataset.data['targets'].to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                accuracy = (outputs.argmax(-1) == targets).float().mean()
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, accuracy={accuracy.item():.2%}")

                if accuracy > 0.99:
                    print("  Reached 99% accuracy, stopping early")
                    break

    model.eval()
    return model


def run_analysis(args):
    """Run the complete analysis pipeline."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    p = args.modulus
    device = args.device

    print("=" * 60)
    print("NEUROMAP FULL ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Modulus: {p}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")

    # 1. Create dataset
    print("\n[1/6] Creating dataset...")
    dataset = ModularArithmeticDataset(p=p)
    print(f"  Created {dataset.data['num_examples']} examples")

    # 2. Create or load model
    print("\n[2/6] Setting up model...")
    if args.model_path:
        print(f"  Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model = create_hooked_model(vocab_size=p, device=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = create_hooked_model(vocab_size=p, device=device)
        if not args.skip_training:
            model = train_model(model, dataset, args.train_epochs, device)
        else:
            print("  Using untrained model (demo mode)")

    model.eval()

    # 3. Extract embeddings
    print("\n[3/6] Extracting embeddings and activations...")
    embeddings = model.get_number_embeddings()
    print(f"  Embedding shape: {embeddings.shape}")

    # Also run with cache to get all activations
    sample_inputs = dataset.data['inputs'][:100].to(device)
    outputs, cache = model.run_with_cache(sample_inputs)
    print(f"  Cached {len(cache)} activation tensors")

    # 4. Fourier analysis
    print("\n[4/6] Analyzing Fourier structure...")
    fourier_analyzer = FourierAnalyzer(p)

    components = fourier_analyzer.extract_fourier_components(embeddings)
    alignment = fourier_analyzer.measure_fourier_alignment(embeddings)
    circular = fourier_analyzer.detect_circular_structure(embeddings)

    print(f"  Dominant frequency: k={components.dominant_frequency}")
    print(f"  Fourier alignment: {alignment.alignment_score:.2%}")
    print(f"  Is Fourier-based: {alignment.is_fourier_based}")
    print(f"  Is circular: {circular['is_circular']}")

    # Generate and save Fourier report
    fourier_report = generate_fourier_report(fourier_analyzer, embeddings)
    with open(output_dir / 'fourier_report.txt', 'w') as f:
        f.write(fourier_report)

    # Save visualizations
    try:
        fig = fourier_analyzer.visualize_fourier_spectrum(
            embeddings, save_path=str(output_dir / 'fourier_spectrum.png')
        )
        fig = fourier_analyzer.visualize_embedding_circle(
            embeddings, save_path=str(output_dir / 'embedding_circle.png')
        )
        print("  Saved visualizations")
    except Exception as e:
        print(f"  Warning: Could not save visualizations: {e}")

    # 5. Causal analysis
    print("\n[5/6] Running causal interventions...")
    patcher = ActivationPatcher(model, device)

    inputs = dataset.data['inputs'][:100].to(device)
    targets = dataset.data['targets'][:100].to(device)
    corrupted = create_corrupted_inputs(inputs, method='shuffle', p=p)

    causal_results = {}
    for layer_name in list(patcher.hook_points.keys())[:5]:  # Top 5 layers
        try:
            result = patcher.mean_ablation(inputs, targets, layer_name)
            causal_results[layer_name] = {
                'accuracy_drop': result.accuracy_drop,
                'loss_increase': result.loss_increase
            }
            print(f"  {layer_name}: accuracy_drop={result.accuracy_drop:.2%}")
        except Exception as e:
            print(f"  {layer_name}: failed ({e})")

    # Save causal results
    with open(output_dir / 'causal_results.json', 'w') as f:
        json.dump(causal_results, f, indent=2)

    # 6. Circuit discovery
    print("\n[6/6] Discovering circuits...")
    discoverer = CircuitDiscoverer(model, device=device)

    try:
        circuit = discoverer.discover_circuit(dataset, n_samples=50)
        circuit_report = generate_circuit_report(circuit)

        with open(output_dir / 'circuit_report.txt', 'w') as f:
            f.write(circuit_report)

        discoverer.export_circuit_diagram(circuit, output_dir / 'circuit.json', format='json')
        print(f"  Found {len(circuit.components)} circuit components")
    except Exception as e:
        print(f"  Warning: Circuit discovery failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    summary = {
        'modulus': p,
        'n_examples': dataset.data['num_examples'],
        'embedding_dim': embeddings.shape[1],
        'fourier_alignment': alignment.alignment_score,
        'is_fourier_based': alignment.is_fourier_based,
        'is_circular': circular['is_circular'],
        'distance_correlation': circular['distance_correlation'],
        'causal_layers_analyzed': len(causal_results),
    }

    print("\nSummary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print("\nFiles generated:")
    for f in output_dir.iterdir():
        print(f"  - {f.name}")

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args)

"""
Complete pipeline test for neural network concept topology visualization.

This script tests the full pipeline from dataset creation through model training
to concept extraction and graph visualization, validating circular structure preservation.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

# Import all components
# Add Dataset directory to Python path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Dataset'))

from dataset import ModularArithmeticDataset
from validation import CircularStructureValidator
from models.transformer import create_model as create_transformer
from models.mamba_model import create_mamba_model
from models.model_utils import ModelTrainer, ModelEvaluator
from analysis.activation_extractor import ActivationExtractor
from analysis.concept_extractors import ClusteringExtractor
from graph.concept_graph import ConceptGraph
from graph.layout_algorithms import CircularLayout, ForceDirectedLayout
from visualization.basic_viz import ConceptVisualizer
from visualization.graph_metrics import GraphValidator


def test_complete_pipeline(p: int = 7, device: str = 'cpu', verbose: bool = True):
    """
    Test the complete neural topology visualization pipeline.
    
    Args:
        p: Modulus for modular arithmetic
        device: Device to run on
        verbose: Whether to print progress
        
    Returns:
        Dictionary with pipeline results
    """
    results = {}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"NEURAL TOPOLOGY VISUALIZATION PIPELINE TEST")
        print(f"Modular arithmetic: mod {p}")
        print(f"{'='*60}")
    
    # Step 1: Create Dataset
    if verbose:
        print(f"\n1. Creating modular arithmetic dataset (p={p})...")
    
    dataset = ModularArithmeticDataset(p=p, representation='embedding')
    if verbose:
        print(f"   Dataset created: {dataset.data['num_examples']} examples")
        print(f"   Input shape: {dataset.data['inputs'].shape}")
        print(f"   Target shape: {dataset.data['targets'].shape}")
    
    results['dataset'] = {
        'p': p,
        'num_examples': dataset.data['num_examples'],
        'input_shape': list(dataset.data['inputs'].shape),
        'target_shape': list(dataset.data['targets'].shape)
    }
    
    # Step 2: Train Models
    if verbose:
        print(f"\n2. Training models...")
    
    models_results = {}
    
    for model_name, model_creator in [('transformer', create_transformer), ('mamba', create_mamba_model)]:
        if verbose:
            print(f"\n   Training {model_name} model...")
        
        # Create model
        if model_name == 'transformer':
            model = model_creator(vocab_size=p, device=device)
        else:
            model = model_creator(vocab_size=p, device=device)
        
        # Create trainer
        trainer = ModelTrainer(model, device=device, learning_rate=1e-3)
        
        # Create simple train/val split
        total_size = dataset.data['num_examples']
        train_size = int(total_size * 0.8)
        indices = torch.randperm(total_size)
        
        train_inputs = dataset.data['inputs'][indices[:train_size]]
        train_targets = dataset.data['targets'][indices[:train_size]]
        val_inputs = dataset.data['inputs'][indices[train_size:]]
        val_targets = dataset.data['targets'][indices[train_size:]]
        
        # Create data loaders
        train_loader, val_loader = trainer.create_data_loaders(
            train_inputs, train_targets, val_inputs, val_targets, batch_size=16
        )
        
        # Train model (quick training for testing)
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            target_accuracy=0.95,
            verbose=False
        )
        
        # Evaluate
        evaluator = ModelEvaluator(model, device=device)
        eval_results = evaluator.evaluate_on_dataset(dataset)
        
        models_results[model_name] = {
            'final_accuracy': eval_results['overall_accuracy'],
            'training_epochs': len(history['train_loss']),
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
            'model': model,
            'evaluator': evaluator
        }
        
        if verbose:
            print(f"     Final accuracy: {eval_results['overall_accuracy']:.3f}")
            print(f"     Training epochs: {len(history['train_loss'])}")
    
    results['models'] = {k: {key: val for key, val in v.items() if key not in ['model', 'evaluator']} 
                        for k, v in models_results.items()}
    
    # Step 3: Extract Activations and Concepts
    if verbose:
        print(f"\n3. Extracting activations and concepts...")
    
    concept_results = {}
    
    for model_name, model_data in models_results.items():
        if verbose:
            print(f"\n   Analyzing {model_name}...")
        
        model = model_data['model']
        evaluator = model_data['evaluator']
        
        # Extract number representations
        try:
            number_reps = evaluator.get_number_representations('embeddings')
            if verbose:
                print(f"     Extracted representations: {number_reps.shape}")
        except:
            # Fallback: extract from all inputs
            extractor = ActivationExtractor(model, device=device)
            inputs = torch.stack([torch.arange(p), torch.zeros(p, dtype=torch.long)], dim=1)
            activations = extractor.extract_activations(inputs, ['aggregated'])
            number_reps = activations['aggregated']
            if verbose:
                print(f"     Extracted representations (fallback): {number_reps.shape}")
        
        # Validate circular structure
        validator = CircularStructureValidator(p)
        structure_validation = validator.validate_embeddings(number_reps, visualize=False)
        
        # Extract concepts using clustering
        clusterer = ClusteringExtractor(n_concepts=p, method='kmeans')
        true_labels = torch.arange(p)  # Ground truth labels
        concept_info = clusterer.extract_concepts(number_reps, true_labels)
        
        concept_results[model_name] = {
            'representations': number_reps,
            'structure_validation': structure_validation,
            'concept_info': concept_info,
            'clusterer': clusterer
        }
        
        if verbose:
            score = structure_validation['overall_assessment']['overall_score']
            quality = structure_validation['overall_assessment']['quality_assessment']
            print(f"     Circular structure score: {score:.3f}")
            print(f"     Structure quality: {quality}")
            
            if 'purity' in concept_info:
                print(f"     Concept purity: {concept_info['purity']:.3f}")
    
    # Step 4: Build Concept Graphs
    if verbose:
        print(f"\n4. Building concept graphs...")
    
    graph_results = {}
    
    for model_name, concept_data in concept_results.items():
        if verbose:
            print(f"\n   Building graph for {model_name}...")
        
        representations = concept_data['representations']
        true_labels = torch.arange(p)
        
        # Create concept graph
        concept_graph = ConceptGraph(
            concept_representations=representations,
            true_labels=true_labels
        )
        
        # Build graph using multiple methods
        methods = ['knn', 'circular']
        graph_results[model_name] = {}
        
        for method in methods:
            if method == 'knn':
                graph = concept_graph.build_graph(method='knn', k=2)
            else:
                graph = concept_graph.build_graph(method='circular')
            
            # Compute layout
            if method == 'circular':
                layout_algo = CircularLayout()
                positions = layout_algo.compute_layout(graph, true_labels=true_labels)
            else:
                layout_algo = ForceDirectedLayout(iterations=100)
                positions = layout_algo.compute_layout(graph)
            
            # Validate graph
            graph_validator = GraphValidator('circular')
            validation_results = graph_validator.validate_graph(
                graph, concept_graph.distance_matrix, positions, true_labels, p
            )
            
            graph_results[model_name][method] = {
                'graph': graph,
                'positions': positions,
                'validation': validation_results,
                'concept_graph': concept_graph
            }
            
            if verbose:
                overall_score = validation_results['overall_assessment']['overall_score']
                quality_rating = validation_results['overall_assessment']['quality_rating']
                print(f"     {method} graph - Score: {overall_score:.3f}, Quality: {quality_rating}")
    
    results['concepts'] = {k: {key: val for key, val in v.items() if key not in ['representations', 'clusterer']} 
                          for k, v in concept_results.items()}
    results['graphs'] = {k: {method: {key: val for key, val in method_data.items() 
                                    if key not in ['graph', 'concept_graph']} 
                            for method, method_data in v.items()} 
                        for k, v in graph_results.items()}
    
    # Step 5: Create Visualizations
    if verbose:
        print(f"\n5. Creating visualizations...")
    
    visualizer = ConceptVisualizer(figsize=(12, 8))
    
    # Create output directory
    output_dir = Path('test_output')
    output_dir.mkdir(exist_ok=True)
    
    # Visualize embeddings comparison
    representations_dict = {name: data['representations'] 
                          for name, data in concept_results.items()}
    
    try:
        visualizer.compare_model_representations(
            representations_dict, 
            labels=torch.arange(p),
            title=f"Model Representations Comparison (mod {p})",
            save_path=output_dir / 'model_comparison.png'
        )
        if verbose:
            print(f"     Saved model comparison to {output_dir}/model_comparison.png")
    except Exception as e:
        if verbose:
            print(f"     Warning: Could not create model comparison: {e}")
    
    # Visualize concept graphs
    for model_name, model_graphs in graph_results.items():
        for method, graph_data in model_graphs.items():
            try:
                fig = visualizer.visualize_concept_graph(
                    graph_data['graph'],
                    graph_data['positions'],
                    node_labels={i: str(i) for i in range(p)},
                    true_labels=torch.arange(p),
                    title=f"{model_name.capitalize()} - {method.capitalize()} Graph",
                    save_path=output_dir / f'{model_name}_{method}_graph.png',
                    interactive=False
                )
                if verbose:
                    print(f"     Saved {model_name} {method} graph to {output_dir}/{model_name}_{method}_graph.png")
            except Exception as e:
                if verbose:
                    print(f"     Warning: Could not create {model_name} {method} graph: {e}")
    
    # Visualize circular structure for best performing model
    best_model = None
    best_score = -1
    
    for model_name, concept_data in concept_results.items():
        score = concept_data['structure_validation']['overall_assessment']['overall_score']
        if score > best_score:
            best_score = score
            best_model = model_name
    
    if best_model:
        try:
            best_reps = concept_results[best_model]['representations']
            best_validation = concept_results[best_model]['structure_validation']
            
            visualizer.visualize_circular_structure(
                best_reps,
                p,
                best_validation,
                title=f"Circular Structure Analysis - {best_model.capitalize()}",
                save_path=output_dir / f'{best_model}_circular_structure.png'
            )
            if verbose:
                print(f"     Saved circular structure analysis to {output_dir}/{best_model}_circular_structure.png")
        except Exception as e:
            if verbose:
                print(f"     Warning: Could not create circular structure visualization: {e}")
    
    results['visualization'] = {
        'output_directory': str(output_dir),
        'best_model': best_model,
        'best_score': best_score
    }
    
    # Step 6: Summary
    if verbose:
        print(f"\n6. Pipeline Summary:")
        print(f"   Dataset: {dataset.data['num_examples']} examples for mod {p}")
        print(f"   Models trained: {len(models_results)}")
        print(f"   Best performing model: {best_model} (score: {best_score:.3f})")
        
        print(f"\n   Model Accuracies:")
        for name, data in models_results.items():
            print(f"     {name}: {data['final_accuracy']:.3f}")
        
        print(f"\n   Circular Structure Scores:")
        for name, data in concept_results.items():
            score = data['structure_validation']['overall_assessment']['overall_score']
            quality = data['structure_validation']['overall_assessment']['quality_assessment']
            print(f"     {name}: {score:.3f} ({quality})")
        
        print(f"\n   Graph Validation (best scores):")
        for name, model_graphs in graph_results.items():
            best_method_score = -1
            best_method = None
            for method, graph_data in model_graphs.items():
                score = graph_data['validation']['overall_assessment']['overall_score']
                if score > best_method_score:
                    best_method_score = score
                    best_method = method
            print(f"     {name}: {best_method_score:.3f} ({best_method})")
        
        print(f"\n✅ Pipeline test completed successfully!")
        print(f"   Results saved to: {output_dir}")
    
    results['summary'] = {
        'pipeline_success': True,
        'best_model': best_model,
        'best_circular_score': best_score,
        'output_directory': str(output_dir)
    }
    
    return results


def quick_test(p: int = 5):
    """Quick test with minimal output."""
    print(f"Running quick pipeline test (p={p})...")
    
    try:
        results = test_complete_pipeline(p=p, verbose=False)
        
        print(f"✅ Quick test passed!")
        print(f"   Best model: {results['summary']['best_model']}")
        print(f"   Circular score: {results['summary']['best_circular_score']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test neural topology visualization pipeline')
    parser.add_argument('--p', type=int, default=7, help='Modulus for arithmetic')
    parser.add_argument('--device', default='cpu', help='Device to run on')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        success = quick_test(args.p)
        exit(0 if success else 1)
    else:
        try:
            results = test_complete_pipeline(
                p=args.p, 
                device=args.device, 
                verbose=args.verbose
            )
            print(f"\n🎉 Full pipeline test completed successfully!")
        except Exception as e:
            print(f"\n💥 Pipeline test failed: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
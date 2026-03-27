import torch
from typing import Any, Dict, List, Optional
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    MergeMethod, 
    MergeTensorInput, 
    ConfigParameterDef,
)
# Note: Ensure your local mergekit version supports this registration decorator
from mergekit.merge_methods import register_merge_method

class LRPMergeTask(Task[torch.Tensor]):
    """
    Performs per-tensor calculations including delta computation, 
    LRP-based functional trimming, and weighted averaging.
    """
    def run(self, **kwargs) -> torch.Tensor:
        # 1. Loading the base tensor and computing deltas
        base_tensor = kwargs.get("base_tensor")
        model_tensors = kwargs.get("model_tensors", {})  # Dict[ModelReference, Tensor]
        weights = kwargs.get("model_weights", {})        # Dict[ModelReference, float]
        lrp_scores = kwargs.get("lrp_scores", {})        # Dict[ModelReference, Tensor]
        density = kwargs.get("density", 0.01)

        merged_deltas = torch.zeros_like(base_tensor)
        total_weight = sum(weights.values())

        for ref, fine_tuned_weight in model_tensors.items():
            # Calculate Delta (Task Vector): FT - Base [4, 5]
            delta = fine_tuned_weight - base_tensor
            
            # 2. Functional Trimming (Sparsification)
            # Use LRP scores if available, otherwise fallback to magnitude [2, 3]
            importance = lrp_scores.get(ref)
            if importance is None:
                importance = delta.abs() # Magnitude fallback

            # Find threshold for top 'density' percent (e.g., top 1%) [6]
            k = int(density * importance.numel())
            if k > 0:
                threshold = torch.kthvalue(
                    importance.flatten(), 
                    importance.numel() - k + 1
                ).values
                mask = (importance >= threshold).to(delta.dtype)
                sparse_delta = delta * mask
            else:
                sparse_delta = torch.zeros_like(delta)

            # 3. Applying Weighted Parameter Averaging to Deltas [7, 8]
            normalized_lambda = weights.get(ref, 1.0) / total_weight
            merged_deltas += normalized_lambda * sparse_delta

        # Final θ = θ_base + Σ(λ * sparse_delta)
        return base_tensor + merged_deltas

    # Map standard mergekit execution to your requested run() method
    def execute(self, **kwargs) -> torch.Tensor:
        return self.run(**kwargs)

@register_merge_method("lrp")
class LRPMerge(MergeMethod):
    """
    Registers the 'lrp' method and initializes the task graph.
    """
    def name(self) -> str:
        return "lrp"

    def parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="density", default_value=0.01)]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", default_value=1.0)]

    def make_task(
        self,
        *,
        output_weight: Any,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
    ) -> Task:
        # 1. Initialize LRPMergeTask with required clinical research parameters
        return LRPMergeTask(
            base_model_ref=base_model,
            base_tensor=tensors[base_model] if base_model else None,
            model_tensors={m: tensors[m] for m in tensor_parameters.keys()},
            model_weights={m: p.get("weight") for m, p in tensor_parameters.items()},
            density=parameters.get("density"),
            # In a real pipeline, LRP scores would be loaded via the 'tensors' input
            # assuming they were pre-computed and added to the task graph [3].
            lrp_scores={m: tensors.get(f"{m}_lrp") for m in tensor_parameters.keys()}
        )
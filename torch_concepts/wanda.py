
# ------------------------------------------------------------------
# WANDA: differentiable graph generation
# ------------------------------------------------------------------
@torch.no_grad()
def _initialize_wanda_random(
    generator: GraphGeneratorLearnable, _data: Any,
) -> None:
    nn.init.normal_(generator.np_params, std=generator.priority_var)


def _wanda_forward(
    self: GraphGeneratorLearnable, _dataset=None,
) -> torch.Tensor:
    differences = self.np_params.T - self.np_params
    identity = torch.eye(self.n_concepts, device=differences.device)
    adjacency = differences * (1 - identity)

    if not self.hard_threshold:
        return adjacency

    hard_adjacency = (differences > self.threshold).float()
    hard_adjacency = torch.where(
        hard_adjacency.abs() < self.eps,
        torch.zeros_like(adjacency),
        hard_adjacency,
    )
    return adjacency + (hard_adjacency - adjacency).detach()


@GraphGeneratorLearnable.register_source("WANDA", names=["wanda"])
def _load_wanda_source(
    generator: GraphGeneratorLearnable,
    name: str,
    concept_names: List[str],
    priority_var: float = 1.0,
    hard_threshold: bool = True,
    threshold_init: float = 0.0,
    eps: float = 1e-12,
    ) -> GraphGeneratorLearnableSpec:
    if name != "wanda":
        raise ValueError("The WANDA source supports only name='wanda'.")
    if threshold_init < 0:
        raise ValueError("threshold_init must be non-negative.")
    generator.concept_names = list(concept_names)
    generator.n_concepts = len(generator.concept_names)
    generator.np_params = nn.Parameter(
        torch.zeros(generator.n_concepts, 1)
    )
    generator.priority_var = priority_var / math.sqrt(2)
    generator.register_buffer(
        "threshold",
        torch.full((generator.n_concepts,), threshold_init),
    )
    generator.hard_threshold = hard_threshold
    generator.eps = eps
    return GraphGeneratorLearnableSpec(
        forward=_wanda_forward,
        initializations={
            "random": GraphGeneratorInitializationSpec(
                _initialize_wanda_random,
            ),
        },
    )

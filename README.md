# LLM-driven design of physics-constrained constitutive models: two agents are better than one

This repository contains the code and artifacts accompanying the manuscript "LLM-driven design of physics-constrained constitutive models: two agents are better than one" by Tacke et al.

We introduce a multi-agent LLM-driven approach for constitutive model generation: a Creator agent proposes a constitutive model tailored to the data, while an Inspector agent critically audits each proposal against nine physical constraints and returns it for refinement whenever a violation is detected. The framework is demonstrated on constitutive artificial neural networks (CANNs) and benchmarked on brain tissue and rubber as isotropic materials and on porcine skin as a transversely isotropic material with a preferred fiber direction, using two different LLM backbones (Claude Opus 4.7 and Kimi K2.5).

## Repository structure

.<br>
├── Artifacts/   # Generated models, validation results, chat logs, protocols<br>
└── Code/        # Python scripts implementing the Creator–Inspector pipeline<br>

### Code/

Contains the full Python implementation of the Creator–Inspector pipeline, including the agentic loop, the numerical validators for the nine physical constraints, the training and evaluation routines, and the benchmark datasets. The system and user prompts used to instantiate the Creator and Inspector agents are defined in llm_prompts.py.

### Artifacts/

Contains the outputs produced during our experiments, including the generated constitutive model implementations, validation results, full chat transcripts between the Creator and Inspector, and run protocols. The complete agent conversations can be inspected here, complementing the prompt definitions in the code. The artifacts cover all six configurations of the main experiments (Creator alone, Creator with Inspector, and Creator with tool-enabled Inspector, each for both LLM backbones) on all four benchmark problems, as well as the corresponding minimal-prompt ablation on the synthetic rubber dataset (archives 07-12).

## Requirements

The code was developed and tested with Python 3.12.7.

## Citation

If you use this code or build upon this work, please cite our manuscript:

@article{tacke2026llm,<br>
  title   = {LLM-driven design of physics-constrained constitutive models: two agents are better than one},<br>
  author  = {Tacke, Marius and Busch, Matthias and Abdolazizi, Kian and Eichinger, Jonas and Linka, Kevin and Aydin, Roland and Cyron, Christian},<br>
  year    = {2026}<br>
}

## Contact

For questions regarding the code or the manuscript, please contact:

Marius Tacke — marius.tacke@hereon.de<br>
Christian Cyron — christian.cyron@hereon.de<br>


#!/usr/bin/env bash
# File: run_layers.sh
#
# Responsibilities:
# - Run layer-wise probing and attention visualization in one command.
# - Save results to artifacts/probes and artifacts/figures.
#
# Contributors:
# - <Name 1>
# - <Name 2>
# - <Name 3>

python -m src.layers.probe --config config/default.yaml
python -m src.layers.attention_viz --config config/default.yaml
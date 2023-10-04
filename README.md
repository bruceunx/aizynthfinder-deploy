## Modify Aizynthfinder for end-users

- remove training module

- remove other algorithms other than mcts

- fix tying error and refactor context module, etc.

- add script for load stock data to database to reduce memory usage.

## How to use this package

- python >= 3.10

- download trained model from original code base

  - uspto_filter_model.onnx // model for filter policy
  - uspto_model.onnx // model for tree generator
  - uspto_templates.csv.gz // templates data
  - zinc_stock.hdf5 // stock data

- install packages

  - numpy, pandas, matplotlib, rdkit, networkx

- run usage.py to check results.

#### TODO

- [ ] rewrite training module with pytorch
- [ ] modify AI models

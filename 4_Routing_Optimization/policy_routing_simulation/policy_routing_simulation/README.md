# Policy Routing Simulation

Compares two road-cleaning dispatch scenarios:

1. **Current-Reactive**
   - Limited observed grid tiles only
   - Observed once per decision period
   - Dispatch only if observed PM is above threshold
   - Candidate grids are solved with the same OP-style routing solver

2. **Ours-Predictive**
   - Citywide predicted PM over all grid tiles
   - Selects top-k high-priority grids
   - Candidate grids are solved with the same OP-style routing solver

The simulation reports four KPIs:

- **Pollution reduction**: sum of PM over visited grids
- **Exposure reduction**: sum of PM × population over visited grids
- **High-risk hit rate**: cleaned high-risk grids / cleaned grids
- **Route efficiency**: benefit per km

## Run

```bash
cd policy_routing_simulation
python main.py
```

Outputs are saved in:

```text
outputs/
```

## Main parameters

Edit the `Config` class in `main.py`.

Important parameters:

```python
n_tiles = 18000
n_observed_current = 250
threshold_pm = 150
top_k_predictive = 120
max_route_km = 80
```

# Soccer Totals Betting Pipeline

This project scaffolds a data-driven workflow for identifying value in soccer totals markets, inspired by the Reddit bettor you referenced. The stack is intentionally lightweight so you can extend it with richer data feeds or alternate modeling techniques.

## Quick Start

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
2. **Fetch historical odds + match data**
    ```bash
    python -m src.data_fetch
    ```
3. **Prepare the modeling dataset**
    ```bash
    python -m src.prepare_dataset
    ```
4. **Train Poisson goal models**
    ```bash
    python -m src.models
    ```
5. **Generate betting signals**
    ```bash
    python -m src.strategy
    ```
6. **Backtest the workflow**
    ```bash
    python -m src.backtest
    ```

Each script reads shared settings from `src/config.py`. Start by editing the league list and seasons there if you want different competitions.

## Notes

-   Data source: [football-data.co.uk](https://www.football-data.co.uk/) free CSVs. They cover most European top and second divisions, including totals odds.
-   The initial model uses Poisson regression for home and away goal expectations with rolling form features. It is a solid baseline for totals markets and easy to adapt for team totals.
-   Strategy logic focuses on Over/Under 2.5 goals. To extend for other lines or sports, follow the same pattern: compute win probabilities, compare to market odds, size stakes via fractional Kelly.
-   Backtests operate on the same historical data you train with; for out-of-sample evaluation, expand the dataset and adjust the test split in `src/models.py`.

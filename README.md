# policylens

Off-policy evaluation for contextual bandits: given a log of what an existing policy did (action, reward, and the probability it chose that action), estimate how a different policy would have done without actually running it. Implements IPS, self-normalized IPS, the direct method and doubly robust, in plain Python with a JSON CLI.

```bash
python -m policylens evaluate --input examples/log.json
python -m policylens evaluate --input examples/log.json --clip 1
python -m policylens demo
```

Each row in the log needs the action taken, the reward, the logging policy's probabilities over all actions, the new policy's probabilities, and a reward model's prediction for each action:

```json
{"action_names": ["a", "b"], "rows": [
  {"action": 0, "reward": 1, "behavior": [0.5, 0.5],
   "target": [0.25, 0.75], "reward_model": [0.6, 0.4]}]}
```

Alongside the estimates you get importance weight stats (max, effective sample size) and a flag for rows where the new policy picks an action the old one never would, which is where all of these estimators fall apart.

## The demo

Two contexts, two actions. The true value of the new policy is 0.78 (the old one gets 0.29).

| setup | IPS | SNIPS | DM | DR |
|---|---:|---:|---:|---:|
| balanced log, wrong reward model | 0.780 | 0.780 | 0.500 | 0.780 |
| same, weights clipped at 1 | 0.185 | 0.617 | 0.500 | 0.535 |
| 2,000 noisy rows, true reward model | 0.729 | 0.753 | 0.780 | 0.753 |
| same rows, cross-fitted reward model | 0.729 | 0.753 | 0.750 | 0.751 |

The first two rows show the classic tradeoff: clipping makes the weights look healthier and biases the answer. The last two show that even with a perfect reward model, 2,000 noisy rows still don't land on 0.78.

## Cross-fitting

If you don't have reward predictions, `crossfit` makes them from categorical contexts: it splits the rows into folds and predicts each row from the average reward of the same context and action in the other folds, shrunk toward 0.5.

```bash
python -m policylens crossfit --input examples/categorical.json --folds 3 --seed 17
```

It splits by row, so if the same user shows up many times, information can leak between folds.

None of this checks that your logged probabilities are correct, and there are no confidence intervals. DESIGN.md and LEDGER.md have the formulas, and https://midya8.github.io/policylens/ shows each row's contribution.

Tests: `python -m unittest discover -s tests` (39 tests).

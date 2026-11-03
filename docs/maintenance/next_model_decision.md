# C.5.3 validation: economic target and model selection

## Why change the validation target?

The previous binary trade label could be positive only when the upper barrier
was touched: all timeouts were treated as a loss of round-trip costs, even
when price rose before expiration. In the AAPL validation sample, 39 of 50
fully evaluated forecasts were timeouts. That proxy cannot reliably measure
precision of positive-return positions.

## Current definition (research proxy, not executable P&L)

* The entry reference is the first strictly future minute **open** after a
  decision, using the existing timestamp and availability convention.
* Upper barrier: assume a fill at the upper price.
* Lower barrier: use the lower price, or a worse observed bar open if the price
  gaps below the stop.
* Timeout: mark the position to the close of the final horizon bar.
* Subtract the specified fixed round-trip cost from the gross percentage return.
* Exclude incomplete horizons, gaps and same-bar ambiguous dual touches.

This is **not a fill simulator**: high/low touches do not prove a fill at the
barrier, and last-bar close may not be attainable for a market order.
Opportunity selection based on this validation return still risks overfitting.
A later executable simulator must model next tradable exit, slippage and
order mechanics.

## Model comparison

Legacy summed objective remains the benchmark: in the shared 50 AAPL
validation examples, it had lower Brier/log loss. Mean-normalized loss
improved upper-first AUC but did not establish superior economic precision.
Do not change the default from the reported evidence.

## Next research sequence

1. Use the new marked-return diagnostics to evaluate **positive-net precision**
   using unchanged chronological validation examples.
2. Keep probability scoring and binary profitability evaluation separate.
3. Analyze per-session opportunity frequency; 50 decisions (with only five
   observed upper events) provide weak precision estimates.
4. Implement time-consistent execution with next-open timeout liquidation,
   commissions and slippage, then test with independent later dates.
5. Try alternative ranking rules or features only after fixing the economic
   outcome definition. Compare on a new validation period or nested training
   split; do not tune repeatedly against the same ten days.
6. Keep holdout untouched until the entire policy and research protocol are frozen.

No holdout data are read by this patch.

Eurex Margin Calculation — Acceptance Test Spec (sample)
As a risk analyst, I want to calculate initial margin for a cleared derivatives portfolio.

Inputs: positions, price vector, vol surfaces, risk factors, member type
Outputs: IM by product and total; warnings for missing data
Constraints: GDPR, audit trail, performance P95 < 250ms
Negative: missing price vector, invalid member, stale vol surface

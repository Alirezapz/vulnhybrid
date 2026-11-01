# Representation audit addendum (before examining new graph test metrics)

The first extraction retained 256 named AST nodes with leaf text. The audit
found two representation limitations: 33.6% of functions are truncated, and
unnamed operator tokens / AST child-field roles are omitted. For a patch that
only changes > to >=, an otherwise identical named-node graph can be identical;
a deterministic classifier cannot recover information that is absent.

Keep that frontend as an explicit baseline, not as a faithful reproduction of
full compiler AST/CPG methods. Add matched controls before interpreting outcomes:
(1) sensitive-API/ancestor selection at the same cap; (2) operator and child-role
payload with breadth-first selection; (3) both additions; (4) larger 512-node
breadth-first cap. Record exact-input collisions for external before/after pairs
and do not attribute representation loss to the GNN operator alone. Evidence
selection is related to established slicing; operator/field preservation is a
representation correction, not world-first algorithmic novelty.

No new graph test accuracy/F1 was consulted to make this addition. This is an
engineering/coverage-driven extension of the initial protocol; the earlier v1
outcomes were already visible. Retain both protocol files and timestamps.

# c2py23: unmarked variants are silently treated as `default: true`, corrupting "reset to default" dispatch

## Summary

`set_backend(None)` in `bslz4_to_sparse` — documented, and intended, to reset
to the fast SIMD-dispatching `kcb` untranspose backend — actually resets to
the portable scalar (`scal`) backend instead, with no error or warning.
This affects every function generated with more than one variant per group
(`bslz4`, `bslz4_csc`, `bslz4_csc_multi`, times 8 dtype groups each), and
means every fresh import of the package silently ran the ~1.4-2x slower
scalar kernel until worked around (see below). Found while benchmarking
against a percentage-scale operational target and discovering a full-frame
decode was much slower than a tiny single-block frame's per-call cost could
explain — the small-frame test isolated it to a backend selection issue,
not call overhead.

Environment: `c2py23==0.5.4` (PyPI), reproduced with the `.c2py` spec
embedded in `src/bslz4_to_sparse.cpp` in this repo.

## Root cause

`c2py23/parser.py:1101`:

```python
v_default = v.get("default", True)
```

A variant dict that does not specify `"default"` at all is treated as
`default: true` by default. It should default to `false` — a variant
should only be eligible as an auto-selected default if the `.c2py` author
explicitly marked it so.

This silently corrupts `c2py23/generator.py`'s fallback-resolution logic
(lines ~483-497), which — for a group with no CPU-feature-guarded (`when:`)
default — scans all variants and picks **the last one for which
`v.default` is true**:

```python
# Find the last default:true variant for fallback
last_default_vi = -1
last_default_name = ""
for vi, v in enumerate(ol.variants):
    if v.default:
        last_default_vi = vi
        last_default_name = v.name
...
b.emit('    _var_{0}_{1} = {2}; _vname_{0}_{1} = "{3}";'.format(name, gi, last_default_vi, last_default_name))
```

In our spec, each dtype group lists two variants:

```jsonc
"variants": [
    {"sig": "bslz4_u16_kcb(...)", "default": True},
    {"sig": "bslz4_u16_scal(...)"},   // no "default" key at all
]
```

Because of the parser bug, `scal`'s omitted key is read as `default=True`
too. Both variants now satisfy `if v.default:` in the "last one wins" scan,
and since `scal` is listed *after* `kcb`, it wins — silently overriding the
author's explicit `"default": True` on `kcb`.

## Reproduction

Generated `bslz4_to_sparse_wrapper.c` (from this repo's `src/bslz4_to_sparse.cpp`):

```c
static void _resolve_bslz4_1(void) {
    _var_bslz4_1 = 1; _vname_bslz4_1 = "bslz4_u16_scal";   // should be kcb (index 0)
    ...
}
static void _resolve_bslz4(void) {
    _resolve_bslz4_0(); _resolve_bslz4_1(); ... _resolve_bslz4_7();
}
```

`_resolve_bslz4()` is what `_rebind_bslz4(None)` calls — i.e. what
`bslz4_to_sparse.set_backend(None)` (or simply never calling `set_backend`
at all) resolves to. Every one of `_resolve_bslz4_0` through `_7` (and the
identically-structured `_resolve_bslz4_csc_N` / `_resolve_bslz4_csc_multi_N`
for the CSC entry points) hardcodes the scalar variant regardless of which
one was marked `default: true` in the source `.c2py` dict.

Confirmed by direct measurement (single core, real Eiger4M-shape frame,
mu=1.0 Poisson data):

| state | ms/call |
|---|---|
| `set_backend(None)` (buggy default) | ~23.0 |
| `set_backend('scal')` (explicit) | ~23.4 |
| `set_backend('kcb')` (explicit) | ~15.7 |

`None` matches `'scal'`, not `'kcb'`, confirming the misresolution.

A tiny single-block (~4KB) frame at high repetition showed both old
(f2py, no variant concept) and new (c2py23) call overhead is genuinely
microseconds (5.98us vs 2.72us) — ruling out Python/c2py call-boundary
overhead as the cause of the much larger full-frame gap, and pointing
straight at which *kernel* was actually running.

## Why this matters beyond this one bug

We're planning to add compile-time ISA-specific variants (sse2/avx2/avx512,
guarded by `"when": "c2py_amd64_avx512f"` etc., the same mechanism
c2py23 already exposes and that `c2bslz4` uses) for hand-vectorized kernels.
Every such group will need exactly one *unconditional* fallback variant
(e.g. a portable scalar kernel with no `when:` guard) alongside several
CPU-feature-guarded ones. Under the current parser behavior, that
unconditional fallback will always compete for "last unconditional
default" against anything else lacking an explicit `"default": false`,
regardless of intent — this bug is not a one-off, it will keep
recurring for every future multi-variant group unless fixed.

## Suggested fix

- `c2py23/parser.py:1101`: `v.get("default", True)` → `v.get("default", False)`.
- Additionally, `generator.py`'s "find the last default:true variant"
  scan should probably raise (like the existing "no variant has
  default: true" check just below it) if **more than one** variant in a
  group satisfies `v.default`, rather than silently picking the last one —
  an ambiguous default is as much an authoring error as a missing one.

## Workaround applied here

`src/__init__.py` now calls `set_backend("kcb")` unconditionally at import
time, forcing the documented/intended default until this is fixed upstream
in c2py23. See the comment there for detail.

## Maintainer review response

The report is accurate and independently reproduced against both this repo
and `bslz4_to_sparse_llm`:

- `c2py23/parser.py:1101` reads exactly `v.get("default", True)`.
- `c2py23/generator.py:481-497` contains exactly the "last default:true
  variant wins" fallback quoted above.
- `bslz4_to_sparse.cpp` really does declare `kcb` with `"default": True`
  followed by `scal` with no `default` key, for every dtype group.
- The generated wrapper really does resolve `_var_bslz4_1 = 1`
  (`bslz4_u16_scal`) — the bug reproduces exactly as described.

Good diagnosis, and the tiny-frame-vs-full-frame isolation of call overhead
from kernel selection is solid methodology. But the suggested fix is not
safe as written, and the deeper issue is a specification ambiguity, not
just a wrong default value.

### The suggested fix breaks the documented, tested convention

`docs/specification.md:608-610` currently documents today's behavior as
*intentional*: "At least one variant per group must have `default: True`
(**the default**)." The project's own canonical multi-variant example
(`examples/simd_dispatch/polysimd.c2py`) relies on exactly this: it
declares one unconditional fallback variant (`poly_f32_scalar`) with no
`default` key, counting on the implicit `True`.

Flipping `parser.py:1101` to `v.get("default", False)` would break that
example and every other `.c2py` spec written to the documented convention:
their unconditional fallback variant would silently become
`default: False`, and generation would then fail outright on "no variant
has default: true" — the exact check that's supposed to catch this.

The second suggested fix (raise when more than one variant satisfies
`v.default`) is closer, but also too broad as worded. It would misfire on
the legitimate, intended pattern this project is about to lean on more
heavily: several `when:`-guarded variants (avx512, avx2, ...) that are
each implicitly `default: True`, plus one unconditional scalar fallback.
All of those are "default: True" today by design, and that's *not*
ambiguous — only one of them (the unconditional one) is actually
competing for the fallback slot the "last one wins" scan fills in.

The narrower, backward-compatible fix: scope both the tie-break and the
ambiguity check to variants where `when_expr is None`. Raise if more than
one unconditional variant is `default: True`; otherwise there is exactly
one, and it should always be that one, regardless of scan order. Since
`kcb` and `scal` here are both unconditional, this would have raised
loudly at generation time instead of silently picking `scal` — without
touching `parser.py`'s default semantics or the documented single-fallback
convention.

### The real problem: `"default"` is one field doing two jobs

Digging into this, I think the report is right that "this will keep
recurring," but the reason isn't the parser's default value — it's that
`default` in the spec currently means two different things at once:

1. "this variant is eligible for auto-resolve at all" (the opposite of
   `default: false`, which is rebind-only), and
2. "this is *the* anointed fallback to use when nothing else matches."

A single boolean can't distinguish "one of several auto-eligible
CPU-guarded variants" from "the one true fallback." An author who writes
one `"default": True` variant and one bare variant naturally reads that as
"I picked the default" — they have no reason to expect the bare variant
also opted itself into the same pool, competing on undocumented
declaration-order tie-break rules. That's not really a misunderstanding on
their part; the spec doesn't say what they'd need to know to avoid it.

Concretely, I'd want the spec to let an author name the fallback
explicitly and unambiguously — e.g. a group-level `"default_variant":
"bslz4_u16_kcb"`, or a variant-level `"fallback": true` distinct from
`"default"` — rather than deriving it implicitly from scan order over a
field that also governs auto-resolve eligibility.

### `set_backend(None)` / `_rebind_<name>(None)` compounds this

Separately: `_rebind_<name>(None)` re-running "auto-resolve" is itself a
weak contract. Calling `set_backend("kcb")` names an actual variant — it's
unambiguous by construction. Calling `set_backend(None)` names nothing; it
re-derives an answer from a scan whose tie-break isn't part of the
documented contract at all (only "no `when:` match" and "first `when:`
match" are documented — the final unconditional fallback's tie-break
among multiple candidates is not).

If the spec required exactly one explicitly-named fallback per group (see
above), `_rebind(None)` would just be "look up the named fallback,"
full stop — no scan, no ambiguity, nothing for declaration order to
silently decide. Right now `None` is a proxy for "whatever the generator's
internal scan happens to pick," which is a much weaker guarantee than the
API's docstring ("re-run auto-resolve") implies, and than callers coding
against it would assume.

### Suggested path forward

- Keep `parser.py:1101`'s default as `True` — don't break the documented
  convention or the existing example/tests.
- Add the ambiguity check, scoped to `when_expr is None` variants only.
- Longer term, consider an explicit fallback-naming mechanism in the spec
  (`default_variant:` at the group level, or similar) so `_rebind(None)`
  resolves to a name the author actually wrote down, not a derived one.
  This also gives `_variants_<name>()` a natural way to report which
  variant is *the* default versus merely auto-eligible.

### Update: confirmed more severely once a `when`-guarded variant is added

We since converted this project to c2py23's `expand` mechanism (one
Python function per dtype instead of one polymorphic function per
family) and added a third, CPU-feature-guarded variant (`sse`, `"when":
"c2py_amd64_sse2"`) alongside the existing `kcb` (`"default": True`) and
`scal` (no `default` key). The bug doesn't just misresolve to the wrong
*unconditional* variant anymore -- with a `when`-guarded variant in the
mix, `kcb` becomes **completely unreachable** via `None`/auto-resolve on
any ordinary x86-64 machine:

```c
static void _resolve_bslz4_u16_0(void) {
    if (c2py_amd64_sse2) {           // sse's default=True (bug) + when-guard
        _var_bslz4_u16_0 = 1; ...    // picked here, every x86-64 machine
        return;
    }
    _var_bslz4_u16_0 = 2; ...        // falls to scal (last unconditional
                                      // default, same bug as before) if not
}
```

Since SSE2 is x86-64 baseline, `c2py_amd64_sse2` is true on essentially
every real machine, so this branch is taken first and `kcb` -- despite
being the one variant actually marked `default: True` by us -- is never
reached by the fallthrough at all. This is exactly the failure mode your
proposed fix (scope the tie-break/ambiguity check to `when_expr is None`
variants) would catch and prevent: `kcb` and `scal` are the two
unconditional variants competing for the fallback slot; `sse`'s
`when`-guarded eligibility is orthogonal and shouldn't be conflated with
it. Not blocking for us (`set_backend('kcb')` targets the variant name
directly and bypasses this resolver entirely), but a concrete data point
that this gets worse, not better, as real per-ISA variant lists grow --
worth weighing when you prioritize the fix.

### Resolved

Fixed on branch `variant-fallback-marker` (commit `ed4e9f9`, "Add
fallback: true singleton marker for ambiguous variant groups") --
exactly the scoped-to-`when_expr is None` fix proposed above, plus two
more cases we hadn't spotted (a when-guarded default:true variant
stealing the fallback slot from a single unconditional one regardless
of declaration order; a group with zero unconditional default:true
variants at all). All three are now checked at parse time
(`from_c2py_dict`/`load_c2py`), not just `generate()`, with a clear
`ValueError` naming the group and competing variants instead of a
silent misresolution. Well covered: 7 new regression tests, plus a note
that every existing `.c2py` spec in the c2py23 repo (including the
`polysimd.c2py` example this report's `docs/specification.md` section
is based on) generates byte-identical wrappers before/after.

Verified against this project directly (branch checked out locally,
installed into a scratch venv in place of the PyPI 0.5.4 release):
`_resolve_bslz4_u16_0()` now unconditionally resolves to
`bslz4_u16_kcb` with no `if` checks at all, confirmed by reading the
generated wrapper. This also prompted us to fix our own `.c2py` spec
properly rather than lean on the "fallback" marker: `sse`/`scal` are
now `"default": False` (explicit `set_backend('sse')`/`('scal')` only
-- matching how they were always meant to be used, for benchmarking one
kernel against another, not silent auto-selection), so `kcb` is the
*sole* unconditional `default: True` variant per group and needs no
`"fallback"` marker at all. Full test suite (7 tests) passes unchanged.

Not yet released to PyPI, so `src/__init__.py`'s `set_backend("kcb")`
workaround stays in place until this project's `c2py23` dependency can
be bumped past whatever release includes this fix -- noted there.

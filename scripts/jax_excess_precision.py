"""Example of excess precision creating different results.

The first algorithm is the "eager" mode with proper downcasts to f16.

The second algorithm compiles. Run with different flags to see different results.

```
$ XLA_FLAGS="--xla_allow_excess_precision=true" python scripts/jax_excess_precision.py

[[1.015625 0.      ]
 [0.       1.015625]] float32
[[1.015686 0.      ]
 [0.       1.015686]] float32
```

and

```
$ XLA_FLAGS="--xla_allow_excess_precision=false" python scripts/jax_excess_precision.py

[[1.015625 0.      ]
 [0.       1.015625]] float32
[[1.015625 0.      ]
 [0.       1.015625]] float32
```
"""

import sys

import jax
import jax.numpy as jnp


def main():
    def f(x):
        return jnp.dot(x, x) * jnp.float32(1.0)

    finfo = jnp.finfo("bfloat16")
    x = (jnp.bfloat16(1) + finfo.eps) * jnp.eye(2, dtype="bfloat16")
    print(f(x), f(x).dtype)
    print(jax.jit(f)(x), jax.jit(f)(x).dtype)


if __name__ == "__main__":
    sys.exit(main())

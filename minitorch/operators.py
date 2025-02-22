"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of x."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Return the natural logarithm of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Return the exponential of x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Return the inverse of x."""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Return the derivative of log times a second arg"""
    return y / x


def inv_back(x: float, y: float) -> float:
    """Return the derivative of inv times a second arg"""
    return -y / (x * x)


def relu_back(x: float, y: float) -> float:
    """Return the derivative of ReLU times a second arg"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[Any], Any], iters: Iterable[Any]) -> Iterable[Any]:
    """Map a function over an iterable."""
    return [fn(x) for x in iters]


def zipWith(
    fn: Callable[[Any, Any], Any], xs: Iterable[Any], ys: Iterable[Any]
) -> Iterable[Any]:
    """Zip two iterables together with a function."""
    return [fn(x, y) for x, y in zip(xs, ys)]


def reduce(fn: Callable[[Any, Any], Any], iters: Iterable[Any]) -> Any:
    """Reduce an iterable with a function."""
    list_iters = list(iters)
    if len(list_iters) == 0:
        raise ValueError("reduce() of empty sequence with no initial value")
    elem = list_iters[0]
    i = 1
    while i < len(list_iters):
        elem = fn(elem, list_iters[i])
        i += 1
    return elem


def negList(xs: Iterable[float]) -> Iterable[float]:
    """Negate a list."""
    return map(neg, xs)


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Add two lists together."""
    return zipWith(add, xs, ys)


def sum(xs: Iterable[float]) -> float:
    """Sum a list."""
    list_xs = list(xs)
    if len(list_xs) == 0:
        return 0.0
    return reduce(add, list_xs)


def prod(xs: Iterable[float]) -> float:
    """Product of a list."""
    list_xs = list(xs)
    if len(list_xs) == 0:
        return 1.0
    return reduce(mul, list_xs)

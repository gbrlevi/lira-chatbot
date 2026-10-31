"""Credit card installment rules from the store policy (section 3.1).

The policy has a general table (minimum of R$ 100,00) and a more specific section with
tiers. The specific section prevails, and "exceto abaixo de R$ 50,00" is read as a
R$ 50,00 minimum for up to 3 installments. Both assumptions are documented in the README.
"""

from dataclasses import dataclass

MAX_INSTALLMENTS = 12

# (highest installment count of the tier, minimum value of each installment in cents)
_TIERS = ((3, 5_000), (6, 8_000), (12, 10_000))


@dataclass(frozen=True)
class Installment:
    count: int
    value_cents: int


@dataclass(frozen=True)
class InstallmentPlan:
    total_cents: int
    options: tuple[Installment, ...]

    @property
    def max_count(self) -> int:
        return self.options[-1].count

    def allows(self, count: int) -> bool:
        return any(option.count == count for option in self.options)


def _minimum_for(count: int) -> int:
    return next(minimum for limit, minimum in _TIERS if count <= limit)


def _value_of_each(total_cents: int, count: int) -> int:
    return (2 * total_cents + count) // (2 * count)


def simulate(total_cents: int) -> InstallmentPlan:
    """List every allowed installment count. Paying once on credit is always allowed."""
    if total_cents <= 0:
        raise ValueError("O valor precisa ser positivo.")
    options = [Installment(1, total_cents)]
    for count in range(2, MAX_INSTALLMENTS + 1):
        # Compares totals, because the rounded installment could cross a minimum by a cent.
        if total_cents >= count * _minimum_for(count):
            options.append(Installment(count, _value_of_each(total_cents, count)))
    return InstallmentPlan(total_cents=total_cents, options=tuple(options))

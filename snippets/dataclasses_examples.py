import dataclasses
from dataclasses import dataclass, astuple, fields


@dataclass
class P:
    value: float


@dataclass
class Params:
    a: int
    c: float
    b: str
    p: P


def dataclasses_examples():
    p1 = Params(1, 0.1, "1", P(1.1))
    p2 = Params(2, 0.2, "2", P(1.2))
    p3 = Params(3, 0.3, "3", P(1.3))

    # astyple makes deepcopy
    for e1, e2, e3 in zip(astuple(p1), astuple(p2), astuple(p3)):
        if isinstance(e1, P):
            e1 = P(-e1.value)
            e2.value += 1
            e3.value += 1
    for e1, e2, e3 in zip(astuple(p1), astuple(p2), astuple(p3)):
        print(f"{e1=} {e2=} {e3=}")

    # this will do the trick
    for field in fields(p1):
        e1, e2, e3 = [getattr(o, field.name) for o in [p1, p2, p3]]
        if isinstance(e1, P):
            e1.value += 1
            e2.value += 1
            e3.value += 1
    for e1, e2, e3 in zip(astuple(p1), astuple(p2), astuple(p3)):
        print(f"{e1=} {e2=} {e3=}")


if __name__ == '__main__':
    dataclasses_examples()

class Container:
    value: int

    def __init__(self, value: int):
        self.value = value


class Composition:
    container: Container

    def __init__(self, container):
        self.c = container


def modify_values(v1: float, v2: Container) -> None:
    v1 += 1
    v2.value += 1


def ref_or_value():
    values = {"v1": 0, "v2": Container(0)}

    print(f"{values['v1']=} {values['v2'].value=}")
    modify_values(values["v1"], values["v2"])
    print(f"{values['v1']=} {values['v2'].value=}")
    modify_values(values["v1"], values["v2"])
    print(f"{values['v1']=} {values['v2'].value=}")
    modify_values(values["v1"], values["v2"])
    print(f"{values['v1']=} {values['v2'].value=}")


    container = Container(10)
    compositon = Composition(container)
    print(f"{container.value} {compositon.c.value}")

    container.value += 10
    print(f"{container.value} {compositon.c.value}")


    container.value = 100
    print(f"{container.value} {compositon.c.value}")

    del container
    print(f"{compositon.c.value}")  # python uses reference counting



if __name__ == '__main__':
    ref_or_value()

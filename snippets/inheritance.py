import abc


class Reader(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def read(self) -> str: ...


class FooReader(Reader):
    def read(self) -> str:
        return "foo"


@Reader.register
class BarReader:
    def read(self) -> str:
        return "bar"


assert isinstance(FooReader(), Reader)
assert isinstance(BarReader(), Reader)


class Dog:
    def bork(self) -> None: ...


dog = Dog()
dog.bork()

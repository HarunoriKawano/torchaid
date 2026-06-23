from typing import Protocol, Iterator, TypeVar

T = TypeVar('T')

class SizedIterable(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterator[T]:
        ...

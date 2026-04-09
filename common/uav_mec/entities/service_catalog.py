from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig


@dataclass(slots=True)
class ServiceCatalog:
    service_sizes_bits: tuple[int, ...]

    @classmethod
    def from_config(cls, config: SystemConfig) -> "ServiceCatalog":
        return cls(service_sizes_bits=config.service_size_bits)

    def get_fetch_size_bits(self, service_type: int) -> int:
        return int(self.service_sizes_bits[service_type])

    @property
    def num_service_types(self) -> int:
        return len(self.service_sizes_bits)

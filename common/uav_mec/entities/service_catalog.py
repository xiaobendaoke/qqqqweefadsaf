"""服务目录模块。

该模块维护服务类型到服务大小的映射，
用于估计 UAV 缓存命中、服务拉取时延和协同传输开销。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SystemConfig


@dataclass(slots=True)
class ServiceCatalog:
    """维护服务类型与其对应模型/镜像大小的映射。"""

    service_sizes_bits: tuple[int, ...]

    @classmethod
    def from_config(cls, config: SystemConfig) -> "ServiceCatalog":
        """从配置中的服务大小表构造目录。"""
        return cls(service_sizes_bits=config.service_size_bits)

    def get_fetch_size_bits(self, service_type: int) -> int:
        """返回目标服务被拉取到 UAV 所需的比特数。"""
        return int(self.service_sizes_bits[service_type])

    @property
    def num_service_types(self) -> int:
        return len(self.service_sizes_bits)

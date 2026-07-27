# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ExecutionBackend construction kept outside the future Task Controller."""

from collections.abc import Mapping
from typing import Any

from .base import ExecutionBackend
from .local import LocalExecutionBackend


def create_execution_backend(config: Mapping[str, Any]) -> ExecutionBackend:
    """Create the configured experiment-scoped Backend implementation."""

    kind = config.get("kind") if isinstance(config, Mapping) else None
    if kind == "local":
        return LocalExecutionBackend.from_config(config)
    raise ValueError(f"Unsupported execution_backend.kind: {kind!r}")

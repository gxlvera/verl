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

from __future__ import annotations

import base64
import io
import mimetypes
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from ..client_sdk.errors import AgentServiceProtocolError
from ..client_sdk.executor import AgentExecutor, as_completed
from ..client_sdk.models import TaskId, TaskSnapshot, TaskSpec, TaskStatus


def _select(config: Any, path: str, default: Any = None) -> Any:
    value = config
    for part in path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return default
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return default
    return value


def _plain(value: Any) -> Any:
    """Convert common verl/OmegaConf sample values to JSON-compatible Python values."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items() if str(key) != "_target_"}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"Cannot convert {type(value).__name__} to an Agent Service wire value")


class _TaskWireEncoder:
    """Encode one sample to JSON while keeping binary media inline."""

    def encode(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Image.Image):
            return self._encode_pil_image(value)
        if isinstance(value, bytes | bytearray | memoryview):
            return self._encode_bytes(bytes(value), kind="binary")
        if isinstance(value, torch.Tensor):
            return self.encode(value.detach().cpu().tolist())
        if isinstance(value, np.ndarray):
            return self.encode(value.tolist())
        if isinstance(value, Mapping):
            return self._encode_mapping(value)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [self.encode(item) for item in value]
        if hasattr(value, "item"):
            return self.encode(value.item())
        raise TypeError(f"Cannot convert {type(value).__name__} to an Agent Service wire value")

    def _encode_mapping(self, value: Mapping[Any, Any]) -> dict[str, Any]:
        media_kind = value.get("type")
        if media_kind in {"image", "video", "audio"}:
            encoded_media = self._encode_media_part(value, str(media_kind))
            if encoded_media is not None:
                return encoded_media
        return {str(key): self.encode(item) for key, item in value.items()}

    def _encode_media_part(self, value: Mapping[Any, Any], kind: str) -> dict[str, Any] | None:
        payload_key = kind
        payload = value.get(payload_key)
        raw_bytes = value.get("bytes")
        source_path = value.get("path")

        if isinstance(raw_bytes, list) and all(isinstance(item, int) and 0 <= item <= 255 for item in raw_bytes):
            raw_bytes = bytes(raw_bytes)

        encoded_payload = None
        if isinstance(raw_bytes, bytes | bytearray | memoryview):
            encoded_payload = self._encode_bytes(
                bytes(raw_bytes),
                kind=kind,
                source_path=source_path if isinstance(source_path, str | os.PathLike) else None,
                image=payload if isinstance(payload, Image.Image) else None,
            )
        elif isinstance(payload, Image.Image):
            encoded_payload = self._encode_pil_image(payload, kind=kind)
        elif isinstance(payload, bytes | bytearray | memoryview):
            encoded_payload = self._encode_bytes(bytes(payload), kind=kind)
        elif isinstance(payload, Mapping):
            nested_bytes = payload.get("bytes")
            nested_path = payload.get("path")
            if isinstance(nested_bytes, bytes | bytearray | memoryview):
                encoded_payload = self._encode_bytes(
                    bytes(nested_bytes),
                    kind=kind,
                    source_path=nested_path if isinstance(nested_path, str | os.PathLike) else None,
                )
        elif isinstance(payload, str | os.PathLike):
            payload_text = os.fspath(payload)
            if not payload_text.startswith(("http://", "https://", "data:")):
                path = Path(payload_text)
                if path.is_file():
                    encoded_payload = self._encode_bytes(path.read_bytes(), kind=kind, source_path=path)

        if encoded_payload is None:
            return None

        encoded = {
            str(key): self.encode(item) for key, item in value.items() if key not in {"bytes", "path", payload_key}
        }
        encoded["type"] = kind
        encoded[payload_key] = encoded_payload
        return encoded

    def _encode_pil_image(self, image: Image.Image, *, kind: str = "image") -> dict[str, Any]:
        output = io.BytesIO()
        image_format = (image.format or "PNG").upper()
        if image_format not in Image.SAVE:
            image_format = "PNG"
        image.save(output, format=image_format)
        content_type = Image.MIME.get(image_format, f"image/{image_format.lower()}")
        return self._inline_bytes(
            output.getvalue(),
            kind=kind,
            content_type=content_type,
            metadata={"width": image.width, "height": image.height, "format": image_format},
        )

    def _encode_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        source_path: str | os.PathLike[str] | None = None,
        image: Image.Image | None = None,
    ) -> dict[str, Any]:
        filename = os.path.basename(os.fspath(source_path)) if source_path is not None else None
        content_type = mimetypes.guess_type(filename)[0] if filename else None
        metadata: dict[str, Any] = {}
        if filename:
            metadata["filename"] = filename
        if image is not None:
            metadata.update({"width": image.width, "height": image.height})
            if image.format:
                metadata["format"] = image.format
                content_type = content_type or Image.MIME.get(image.format.upper())
        return self._inline_bytes(
            data,
            kind=kind,
            content_type=content_type or "application/octet-stream",
            metadata=metadata,
        )

    def _inline_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        content_type: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "kind": kind,
            "content_type": content_type,
            "encoding": "base64",
            "data": base64.b64encode(data).decode("ascii"),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        return payload


def _mapping(value: Any, field_name: str, *, allow_none: bool = False) -> dict[str, Any] | None:
    value = _plain(value)
    if value is None and allow_none:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping" + (" or null" if allow_none else ""))
    return dict(value)


def _object_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _token_ids(value: Any, field_name: str) -> list[int]:
    value = _plain(value)
    if not isinstance(value, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in value):
        raise AgentServiceProtocolError(f"trajectory.{field_name} must be an array of integer token IDs")
    return value


@dataclass
class _TrajectoryTensors:
    prompts: torch.Tensor
    responses: torch.Tensor
    response_mask: torch.Tensor
    attention_mask: torch.Tensor
    input_ids: torch.Tensor
    response_logprobs: torch.Tensor | None
    loss_weight: torch.Tensor | None
    routed_experts: torch.Tensor | None
    num_turns: int
    metrics: dict[str, Any]
    extra_fields: dict[str, Any]


class RolloutAdapter:
    """Bridge verl ``DataProto`` batches to Agent Service Task APIs.

    The built-in result decoder accepts the token-level trajectory schema used
    by verl's native AgentLoopOutput: ``prompt_ids``, ``response_ids``,
    ``response_mask`` (or ``loss_mask``), and optional ``response_logprobs`` and
    ``routed_experts``.
    """

    def __init__(self, executor: AgentExecutor, *, config: Any, tokenizer: Any, processor: Any = None):
        self.executor = executor
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.in_flight: set[TaskId] = set()
        self._input_items: dict[TaskId, Any] = {}

    def build_task_spec(self, item: Any) -> TaskSpec:
        task_config = _mapping(_select(self.config, "agent_service.task", {}), "agent_service.task")
        assert task_config is not None

        wire_encoder = _TaskWireEncoder()
        raw_prompt_value = item.non_tensor_batch.get("raw_prompt")
        raw_prompt = wire_encoder.encode(raw_prompt_value) if raw_prompt_value is not None else None
        sample_fields = {
            str(key): wire_encoder.encode(value)
            for key, value in item.non_tensor_batch.items()
            if key not in {"raw_prompt", "__do_sample__"}
        }

        problem = _mapping(task_config.get("problem", {}), "agent_service.task.problem")
        agent = _mapping(task_config.get("agent", {}), "agent_service.task.agent")
        execution = _mapping(task_config.get("execution", {}), "agent_service.task.execution")
        environment = _mapping(task_config.get("environment"), "agent_service.task.environment", allow_none=True)
        reward = _mapping(task_config.get("reward", {}), "agent_service.task.reward")
        lifecycle = _mapping(task_config.get("lifecycle"), "agent_service.task.lifecycle", allow_none=True)
        assert problem is not None and agent is not None and execution is not None and reward is not None

        if raw_prompt is not None:
            # Per-sample input always wins over static config. This is the same
            # list of message dicts a native AgentLoop receives as raw_prompt.
            problem["messages"] = raw_prompt
        elif item.batch is not None and "prompts" in item.batch:
            prompt_ids = item.batch["prompts"]
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            unpadded_ids = prompt_ids[prompt_ids != pad_token_id].detach().cpu().tolist()
            problem["messages"] = [
                {"role": "user", "content": self.tokenizer.decode(unpadded_ids, skip_special_tokens=True)}
            ]
        else:
            raise ValueError("Cannot build problem.messages: sample has neither raw_prompt nor prompts")

        metadata = _mapping(problem.get("metadata", {}), "agent_service.task.problem.metadata")
        assert metadata is not None
        for key in ("data_source", "uid", "index"):
            if key in sample_fields:
                metadata.setdefault(key, sample_fields[key])
        if metadata:
            problem["metadata"] = metadata

        if "agent_name" in sample_fields:
            # Match AgentLoopWorker routing semantics: the dataset-provided
            # per-sample agent_name selects the agent loop, while config is
            # only the default used when the batch has no agent_name field.
            agent["name"] = sample_fields["agent_name"]

        sample_reward = sample_fields.get("reward_model")
        if sample_reward is not None:
            sample_verifier = _mapping(sample_reward, "sample_fields.reward_model")
            configured_verifier = _mapping(reward.get("verifier", {}), "agent_service.task.reward.verifier")
            assert sample_verifier is not None and configured_verifier is not None
            reward["verifier"] = {**sample_verifier, **configured_verifier}

        rollout_config = _select(self.config, "actor_rollout_ref.rollout")
        validate = bool(item.meta_info.get("validate", False))
        sampling_config = _select(rollout_config, "val_kwargs") if validate else rollout_config
        generation = {
            "temperature": _plain(_select(sampling_config, "temperature", 1.0)),
            "top_p": _plain(_select(sampling_config, "top_p", 1.0)),
            "top_k": _plain(_select(sampling_config, "top_k", -1)),
            "max_new_tokens": _plain(_select(rollout_config, "response_length")),
        }
        configured_generation = _mapping(task_config.get("generation", {}), "agent_service.task.generation")
        assert configured_generation is not None
        generation.update(configured_generation)
        trajectory_selection = _mapping(
            task_config.get("trajectory_selection", {"strategy": "longest", "config": {}}),
            "agent_service.task.trajectory_selection",
        )
        assert trajectory_selection is not None

        do_sample_by_default = _select(sampling_config, "do_sample", _select(rollout_config, "do_sample", True))
        if not bool(do_sample_by_default):
            generation.update({"temperature": 0, "top_p": 1.0, "top_k": -1})

        do_sample = item.non_tensor_batch.get("__do_sample__")
        if do_sample is not None and not bool(_plain(do_sample)):
            generation.update({"temperature": 0, "top_p": 1.0, "top_k": -1})

        priority = sample_fields.get("priority")
        if priority is not None:
            lifecycle = dict(lifecycle or {})
            lifecycle.setdefault("priority", int(priority))

        return TaskSpec(
            problem=problem,
            agent=agent,
            execution=execution,
            environment=environment,
            reward=reward,
            generation=generation,
            lifecycle=lifecycle,
            sample_fields=sample_fields,
            trajectory_selection=trajectory_selection,
        )

    def submit_batch(self, batch: DataProto, config: Any = None) -> list[tuple[int, TaskId]]:
        if config is not None:
            self.config = config
        indexed_task_ids = []
        for index in range(len(batch)):
            item = batch[index]
            task_id = self.executor.submit(self.build_task_spec(item))
            if task_id in self._input_items:
                raise AgentServiceProtocolError(f"Agent Service returned duplicate Task ID {task_id}")
            indexed_task_ids.append((index, task_id))
            self.in_flight.add(task_id)
            self._input_items[task_id] = item
        return indexed_task_ids

    def wait_batch(self, indexed_task_ids: Sequence[tuple[int, TaskId]], timeout: float) -> DataProto:
        position_by_task_id = {task_id: index for index, task_id in indexed_task_ids}
        if len(position_by_task_id) != len(indexed_task_ids):
            raise AgentServiceProtocolError("indexed_task_ids contains duplicate Task IDs")
        snapshots: list[TaskSnapshot | None] = [None] * len(indexed_task_ids)

        for snapshot in as_completed(
            executor=self.executor,
            task_ids=position_by_task_id,
            timeout=timeout,
        ):
            position = position_by_task_id[snapshot.task_id]
            snapshots[position] = snapshot
            self.in_flight.discard(snapshot.task_id)

        completed = [snapshot for snapshot in snapshots if snapshot is not None]
        if len(completed) != len(snapshots):
            raise AgentServiceProtocolError("WaitAny completed without returning every requested Task")
        input_items = [self._input_items[task_id] for _, task_id in indexed_task_ids]
        try:
            return self.build_verl_dataproto(completed, input_items)
        finally:
            for _, task_id in indexed_task_ids:
                self._input_items.pop(task_id, None)

    def cancel_in_flight(self) -> None:
        first_error: BaseException | None = None
        for task_id in tuple(self.in_flight):
            try:
                self.executor.cancel(task_id)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            finally:
                self.in_flight.discard(task_id)
                self._input_items.pop(task_id, None)
        if first_error is not None:
            raise first_error

    def build_verl_dataproto(self, snapshots: Sequence[TaskSnapshot], input_items: Sequence[Any]) -> DataProto:
        if len(snapshots) != len(input_items):
            raise AgentServiceProtocolError("Task snapshots and input batch sizes do not match")
        trajectories: list[_TrajectoryTensors] = []
        expanded_snapshots: list[TaskSnapshot] = []
        expanded_input_items: list[Any] = []
        for snapshot, input_item in zip(snapshots, input_items, strict=True):
            selected = self._decode_snapshot(snapshot)
            trajectories.extend(selected)
            expanded_snapshots.extend([snapshot] * len(selected))
            expanded_input_items.extend([input_item] * len(selected))
        if not trajectories:
            raise AgentServiceProtocolError("Agent Service returned no selected training trajectories")

        prompts = torch.stack([trajectory.prompts for trajectory in trajectories])
        responses = torch.stack([trajectory.responses for trajectory in trajectories])
        response_mask = torch.stack([trajectory.response_mask for trajectory in trajectories])
        attention_mask = torch.stack([trajectory.attention_mask for trajectory in trajectories])
        input_ids = torch.stack([trajectory.input_ids for trajectory in trajectories])
        position_ids = compute_position_id_with_mask(attention_mask)

        tensor_values = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

        response_logprobs = [trajectory.response_logprobs for trajectory in trajectories]
        if any(value is not None for value in response_logprobs):
            if not all(value is not None for value in response_logprobs):
                raise AgentServiceProtocolError("response_logprobs must be present for either all or no trajectories")
            tensor_values["rollout_log_probs"] = torch.stack(response_logprobs)  # type: ignore[arg-type]

        loss_weights = [trajectory.loss_weight for trajectory in trajectories]
        if any(value is not None for value in loss_weights):
            if not all(value is not None for value in loss_weights):
                raise AgentServiceProtocolError("loss_weight must be present for either all or no trajectories")
            tensor_values["loss_weight"] = torch.stack(loss_weights)  # type: ignore[arg-type]

        routed_experts = [trajectory.routed_experts for trajectory in trajectories]
        if any(value is not None for value in routed_experts):
            if not all(value is not None for value in routed_experts):
                raise AgentServiceProtocolError("routed_experts must be present for either all or no trajectories")
            tensor_values["routed_experts"] = torch.stack(routed_experts)  # type: ignore[arg-type]

        rewards = []
        reward_extra_infos = []
        for snapshot in expanded_snapshots:
            reward = snapshot.final_reward
            if reward is None:
                raise AgentServiceProtocolError(f"Succeeded Task {snapshot.task_id} has no scalar final reward")
            rewards.append(reward)
            if isinstance(snapshot.reward, Mapping):
                reward_extra_infos.append(
                    {key: value for key, value in snapshot.reward.items() if key not in {"final_reward", "reward"}}
                )
            else:
                reward_extra_infos.append({})

        # Emit the scalar reward as token-level rm_scores with the full reward on
        # the last valid response token, matching verl's reward-loop convention.
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        response_lengths = attention_mask[:, prompts.shape[1] :].sum(dim=1)
        if torch.any(response_lengths <= 0):
            raise AgentServiceProtocolError("Every succeeded trajectory must contain at least one response token")
        rm_scores[torch.arange(len(trajectories)), response_lengths - 1] = torch.tensor(rewards, dtype=torch.float32)
        tensor_values["rm_scores"] = rm_scores

        non_tensor_batch: dict[str, np.ndarray] = {}
        input_keys = set().union(*(item.non_tensor_batch.keys() for item in expanded_input_items))
        for key in input_keys:
            non_tensor_batch[key] = _object_array([item.non_tensor_batch.get(key) for item in expanded_input_items])
        non_tensor_batch["__num_turns__"] = np.array(
            [trajectory.num_turns for trajectory in trajectories], dtype=np.int32
        )

        extra_keys = set().union(*(trajectory.extra_fields.keys() for trajectory in trajectories))
        for key in extra_keys:
            non_tensor_batch[key] = _object_array([trajectory.extra_fields.get(key) for trajectory in trajectories])

        reward_extra_keys = sorted(set().union(*(info.keys() for info in reward_extra_infos)))
        for key in reward_extra_keys:
            non_tensor_batch[key] = _object_array([info.get(key) for info in reward_extra_infos])

        return DataProto(
            batch=TensorDict(tensor_values, batch_size=len(trajectories)),
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "metrics": [trajectory.metrics for trajectory in trajectories],
                "reward_extra_keys": reward_extra_keys,
            },
        )

    def _decode_snapshot(self, snapshot: TaskSnapshot) -> list[_TrajectoryTensors]:
        # V0 policy: any FAILED/CANCELLED Task aborts the training step. The RFC
        # leaves accept/drop/resubmit to the driver; a subclass that wants
        # partial batches or resubmission should override wait_batch instead.
        if snapshot.status is not TaskStatus.SUCCEEDED:
            detail = snapshot.error.message if snapshot.error is not None else "no server error detail"
            raise RuntimeError(f"Agent Service Task {snapshot.task_id} ended as {snapshot.status.value}: {detail}")
        trajectory = snapshot.trajectory
        if not isinstance(trajectory, Mapping):
            raise AgentServiceProtocolError(f"Succeeded Task {snapshot.task_id} has no trajectory object")
        if isinstance(trajectory.get("selected_bundle"), Mapping):
            trajectory = trajectory["selected_bundle"]
        if isinstance(trajectory.get("training_trajectory"), Mapping):
            linear_trajectories = [trajectory["training_trajectory"]]
        elif "trajectories" in trajectory:
            linear_trajectories = trajectory["trajectories"]
            if not isinstance(linear_trajectories, list) or not linear_trajectories:
                raise AgentServiceProtocolError(
                    "Selected trajectory bundle must contain at least one linear trajectory"
                )
        else:
            linear_trajectories = [trajectory]
        if any(not isinstance(item, Mapping) for item in linear_trajectories):
            raise AgentServiceProtocolError("Every selected training trajectory must be a JSON object")
        return [self._decode_trajectory(trajectory) for trajectory in linear_trajectories]

    def _decode_trajectory(self, trajectory: Mapping[str, Any]) -> _TrajectoryTensors:
        if not isinstance(trajectory, Mapping):
            raise AgentServiceProtocolError("training trajectory must be a JSON object")

        prompt_ids = _token_ids(trajectory.get("prompt_ids"), "prompt_ids")
        response_ids = _token_ids(trajectory.get("response_ids"), "response_ids")
        response_mask = _token_ids(trajectory.get("response_mask", trajectory.get("loss_mask")), "response_mask")
        if len(response_ids) != len(response_mask):
            raise AgentServiceProtocolError("trajectory.response_ids and response_mask lengths must match")
        if any(mask not in (0, 1) for mask in response_mask):
            raise AgentServiceProtocolError("trajectory.response_mask values must be 0 or 1")

        prompt_length = int(_select(self.config, "actor_rollout_ref.rollout.prompt_length"))
        response_length = int(_select(self.config, "actor_rollout_ref.rollout.response_length"))
        if len(prompt_ids) > prompt_length:
            raise AgentServiceProtocolError(
                f"trajectory prompt has {len(prompt_ids)} tokens, exceeding configured prompt_length={prompt_length}"
            )
        if not response_ids or len(response_ids) > response_length:
            raise AgentServiceProtocolError(
                f"trajectory response must contain 1..{response_length} tokens, got {len(response_ids)}"
            )

        # verl batch layout: prompts are left-padded to prompt_length, responses
        # right-padded to response_length, so input_ids = [pad..prompt|response..pad]
        # and the attention span is contiguous around the prompt/response seam.
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        padded_prompts = torch.full((prompt_length,), pad_token_id, dtype=torch.long)
        padded_prompts[prompt_length - len(prompt_ids) :] = torch.tensor(prompt_ids, dtype=torch.long)
        padded_responses = torch.full((response_length,), pad_token_id, dtype=torch.long)
        padded_responses[: len(response_ids)] = torch.tensor(response_ids, dtype=torch.long)
        padded_response_mask = torch.zeros(response_length, dtype=torch.long)
        padded_response_mask[: len(response_mask)] = torch.tensor(response_mask, dtype=torch.long)
        attention_mask = torch.zeros(prompt_length + response_length, dtype=torch.long)
        attention_mask[prompt_length - len(prompt_ids) : prompt_length + len(response_ids)] = 1
        input_ids = torch.cat([padded_prompts, padded_responses])

        response_logprobs = trajectory.get("response_logprobs", trajectory.get("logprobs"))
        padded_logprobs = None
        if response_logprobs is not None:
            response_logprobs = _plain(response_logprobs)
            if not isinstance(response_logprobs, list) or len(response_logprobs) != len(response_ids):
                raise AgentServiceProtocolError(
                    "trajectory.response_logprobs must contain one number per response token"
                )
            if any(not isinstance(value, int | float) or isinstance(value, bool) for value in response_logprobs):
                raise AgentServiceProtocolError("trajectory.response_logprobs values must be numbers")
            padded_logprobs = torch.zeros(response_length, dtype=torch.float32)
            padded_logprobs[: len(response_logprobs)] = torch.tensor(response_logprobs, dtype=torch.float32)

        loss_weight = trajectory.get("loss_weight")
        padded_loss_weight = None
        if loss_weight is not None:
            loss_weight = _plain(loss_weight)
            if not isinstance(loss_weight, list) or len(loss_weight) != len(response_ids):
                raise AgentServiceProtocolError("trajectory.loss_weight must contain one number per response token")
            if any(not isinstance(value, int | float) or isinstance(value, bool) for value in loss_weight):
                raise AgentServiceProtocolError("trajectory.loss_weight values must be numbers")
            if any(value < 0 for value in loss_weight):
                raise AgentServiceProtocolError("trajectory.loss_weight values must be non-negative")
            padded_loss_weight = torch.zeros(response_length, dtype=torch.float32)
            padded_loss_weight[: len(loss_weight)] = torch.tensor(loss_weight, dtype=torch.float32)

        padded_routed_experts = None
        routed_experts = trajectory.get("routed_experts")
        if routed_experts is not None:
            routed_experts_tensor = torch.as_tensor(_plain(routed_experts), dtype=torch.long)
            if routed_experts_tensor.ndim != 3 or routed_experts_tensor.shape[0] != len(prompt_ids) + len(response_ids):
                raise AgentServiceProtocolError(
                    "trajectory.routed_experts must have shape [prompt_tokens + response_tokens, layers, top_k]"
                )
            padded_routed_experts = torch.zeros(
                (prompt_length + response_length, *routed_experts_tensor.shape[1:]), dtype=torch.long
            )
            prompt_start = prompt_length - len(prompt_ids)
            padded_routed_experts[prompt_start:prompt_length] = routed_experts_tensor[: len(prompt_ids)]
            padded_routed_experts[prompt_length : prompt_length + len(response_ids)] = routed_experts_tensor[
                len(prompt_ids) :
            ]

        metrics = _mapping(trajectory.get("metrics", {}), "trajectory.metrics")
        extra_fields = _mapping(trajectory.get("extra_fields", {}), "trajectory.extra_fields")
        assert metrics is not None and extra_fields is not None
        num_turns = trajectory.get("num_turns", 0)
        if not isinstance(num_turns, int) or isinstance(num_turns, bool) or num_turns < 0:
            raise AgentServiceProtocolError("trajectory.num_turns must be a non-negative integer")

        return _TrajectoryTensors(
            prompts=padded_prompts,
            responses=padded_responses,
            response_mask=padded_response_mask,
            attention_mask=attention_mask,
            input_ids=input_ids,
            response_logprobs=padded_logprobs,
            loss_weight=padded_loss_weight,
            routed_experts=padded_routed_experts,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )

#!/usr/bin/env bash
# GSM8K multi-turn tool-agent RL (restored)
# GRPO | Qwen2.5-3B-Instruct | tool_agent loop | SGLang async | FSDP
#
# Restores the deleted examples/sglang_multiturn GSM8K tool-agent example using
# the legacy BaseTool ("写法 A") tool verl/tools/gsm8k_tool.py + a custom reward
# function with tool-call shaping to avoid the "collapse to 2 turns" failure
# mode (verl #1569 / PR #4998). Continuous Token is enabled by default.
#
# Prep (run once) -- full official split, leakage-guarded (train/test questions disjoint):
#   online : python3 examples/data_preprocess/gsm8k_tool_agent_loop.py
#   offline: python3 examples/data_preprocess/gsm8k_tool_agent_loop.py --raw_dir /path/to/raw_gsm8k
#            (raw_dir holds raw {train,test}.{parquet,jsonl,json} with columns question/answer)
#   -> ~/data/gsm8k/{train,test}.parquet  (agent_name=tool_agent, 7473 train / 1319 test)
#
# Logging: byted-wandb is installed (drop-in `wandb`); on Merlin this prints a reckon
#   tracking link + sends a Feishu notification. project=$PROJECT_NAME run=$EXPERIMENT_NAME.
#
# Run (8x GPU node):
#   bash examples/sglang_multiturn/run_qwen2_5_3b_gsm8k_tool_agent.sh
#
# Useful env knobs:
#   MODEL_PATH               HF id or local path (default Qwen/Qwen2.5-3B-Instruct)
#   ENABLE_CONTINUOUS_TOKEN  enable Continuous Token (default True)
#   CT_MODEL_FAMILY          continuous_token model family (default qwen25)
#   GSM8K_TOOL_SHAPING_BONUS tool-call shaping bonus (default 0.1)
#   TRAIN_BATCH_SIZE / ROLLOUT_N / TOTAL_STEPS / TEST_FREQ

set -xeuo pipefail

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}

ENABLE_CONTINUOUS_TOKEN=${ENABLE_CONTINUOUS_TOKEN:-True}
CT_MODEL_FAMILY=${CT_MODEL_FAMILY:-qwen25}

NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

TOTAL_STEPS=${TOTAL_STEPS:-50}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:--1}

export GSM8K_TOOL_SHAPING_BONUS=${GSM8K_TOOL_SHAPING_BONUS:-0.1}

PROJECT_NAME=${PROJECT_NAME:-gsm8k_tool_agent}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2.5-3b_tool_agent_ct${ENABLE_CONTINUOUS_TOKEN}_$(date +%Y%m%d_%H%M)}

TOOL_CONFIG="$CONFIG_PATH/tool_config/gsm8k_tool_config.yaml"
REWARD_FILE="$PROJECT_DIR/examples/sglang_multiturn/gsm8k_reward_shaping.py"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.continuous_token.enable="${ENABLE_CONTINUOUS_TOKEN}" \
    actor_rollout_ref.rollout.multi_turn.continuous_token.model_family="${CT_MODEL_FAMILY}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.log_val_generations=20 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.total_epochs=1 \
    "$@"

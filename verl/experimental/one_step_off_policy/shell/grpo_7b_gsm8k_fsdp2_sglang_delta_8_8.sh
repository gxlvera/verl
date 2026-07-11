set -x

# One-step-off (disaggregated) GRPO with DELTA weight sync — Qwen 7B, 2 nodes.
#
# Topology: node A = trainer (8 GPUs, FSDP2), node B = rollout (8 GPUs, SGLang).
# Derived from grpo_0.6b_gsm8k_fsdp2_sglang_delta_2_6.sh; the trainer -> rollout
# weight sync only broadcasts parameters that changed since the previous sync.
#
# Launch (Ray cluster across the 2 nodes, then run this script on the head node):
#   node A (head):   ray start --head --port=6379
#   node B:          ray start --address=<nodeA_ip>:6379
#   node A:          bash grpo_7b_gsm8k_fsdp2_sglang_delta_8_8.sh
#
# Weight-sync profiling probes (from the delta-sharded-profile branch) are ON by
# default; disable with PROFILE=0. The env vars are propagated to all Ray
# workers on both nodes via ray runtime_env.

project_name='GRPO'
exp_name='GRPO-Qwen2.5-7b-gsm8k-fsdp2-sglang-one-step-off-delta-8-8'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}

# 1 trainer node + 1 rollout node, 8 GPUs each
TRAIN_NNODES=${TRAIN_NNODES:-1}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-8}

# delta | delta_sharded
DELTA_BACKEND=${DELTA_BACKEND:-delta}

# Weight-sync profiling. The probes treat ANY non-empty value as enabled
# (bool(os.environ.get(...))), so only inject the vars when PROFILE=1.
PROFILE=${PROFILE:-1}
profile_args=()
if [ "${PROFILE}" = "1" ]; then
    profile_args+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_SYNC_PROFILE='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PROFILE_DELTA_SEND='1'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_PROFILE_NCCL_SEND='1'"
    )
fi

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=1152 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=192 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${DELTA_BACKEND}" \
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.delta.encoding=indices \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.nnodes="${TRAIN_NNODES}" \
    trainer.n_gpus_per_node="${TRAIN_GPUS_PER_NODE}" \
    rollout.nnodes="${ROLLOUT_NNODES}" \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE}" \
    "${profile_args[@]}" $@

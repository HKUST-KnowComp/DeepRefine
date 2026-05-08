# Verl适配Refinement Pipeline说明

本文档说明如何将verl训练代码适配到refinement pipeline，实现不包括LLM construction，只有refinement过程的end-to-end RL training。

## 主要修改点

### 1. 创建RefinementInteraction类
**文件**: `verl/interactions/refinement_interaction.py`

- 替代原来的`RAGInteraction`用于refinement流程
- 处理三个阶段的交互：
  1. **Retrieval + Answerable Judgement**: LLM判断当前子图是否足以回答问题
  2. **Error Abduction**: 如果不可回答，分析原因
  3. **Refinement Actions**: 生成KG refinement actions (insert_edge, delete_edge, replace_node)

### 2. 创建Refinement Reward函数
**文件**: `verl/third_party/autograph_r1/refinement_reward.py`

- 替代原来的`deduce_reward.py`用于refinement过程的reward计算
- Reward组成：
  - **Answerable reward**: 1.0如果query变得可回答，0.0否则
  - **Action quality**: 基于actions的有效性和质量
  - 最终reward = answerable_weight * answerable_reward + action_weight * action_reward

### 3. 实现Refinement Engine Call
**文件**: `verl/workers/rollout/sglang_rollout/sglang_rollout.py`

- 实现`_handle_refinement_engine_call`方法
- 在REFINEMENT状态下调用LLM生成refinement相关的输出
- 更新REFINEMENT状态的处理逻辑

## 配置修改

### 1. Interaction配置

#### 1.1 创建Interaction配置文件

已创建配置文件：`config/interaction_config/refinement_interaction_config.yaml`

配置文件内容：
```yaml
interaction:
  - name: "refinement"
    class_name: "verl.interactions.refinement_interaction.RefinementInteraction"
    config:
      max_hops: 5
      max_triple_num: 60
      history_horizon_size: 3
```

#### 1.2 在训练脚本中指定配置文件路径

在训练脚本（如`scripts/autograph-r1/run_*.sh`）中，通过以下参数指定interaction配置文件：

```bash
actor_rollout_ref.rollout.multi_turn.interaction_config_path='config/interaction_config/refinement_interaction_config.yaml'
```

或者在主配置文件中设置：
```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      interaction_config_path: config/interaction_config/refinement_interaction_config.yaml
```

**注意**：配置文件路径是相对于项目根目录的路径。

#### 1.3 完整示例

在训练脚本中的完整配置示例（参考`scripts/autograph-r1/run_qwen2.5-3b_instruct_graph.sh`）：

```bash
python verl/trainer/main_ppo.py \
    ... \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path='config/interaction_config/refinement_interaction_config.yaml' \
    ...
```

### 2. Reward函数配置
在reward配置中使用refinement_reward：

```yaml
custom_reward_function:
  path: verl/third_party/autograph_r1/refinement_reward.py
  name: compute_score
  reward_kwargs:
    answerable_weight: 0.8
    action_weight: 0.2
```

### 3. Rollout配置
确保rollout配置中设置refinement模式：

```yaml
actor_rollout_ref:
  rollout:
    mode: async  # 或 sync
    # ... 其他配置
```

## 数据加载修改

### 需要修改的地方
1. **数据加载逻辑**: 需要加载已构建的KG数据（从pickle文件），而不是从文档构建
2. **数据集类**: 可能需要创建新的数据集类来加载refinement训练数据

### 数据格式
Refinement训练数据应该包含：
- `question`: 问题文本
- `ground_truth`: 正确答案
- `kg_data_path`: 已构建的KG数据路径（pickle文件）
- 其他metadata

## 与benchmarking_graph_refiner.py的对应关系

| benchmarking_graph_refiner.py | verl训练代码 |
|------------------------------|-------------|
| `Reafiner.refine(query)` | `RefinementInteraction` + LLM生成 |
| `_answerable_judgement` | LLM生成`<judge>yes/no</judge>` |
| `_error_abduction` | LLM生成`<abduction>reason</abduction>` |
| `_kg_refinement_action` | LLM生成`<refinement>actions</refinement>` |
| Reward计算 | `refinement_reward.compute_score` |

## 主要区别

### benchmarking_graph_refiner.py (Inference)
- 直接调用`Reafiner.refine()`进行refinement
- 使用固定的LLM进行judgement、abduction和action generation
- 主要用于评估和benchmarking

### verl训练代码 (RL Training)
- LLM学习生成refinement actions
- 通过RL训练优化refinement策略
- 使用reward信号指导学习过程

## 下一步工作

1. **数据加载**: 实现加载已构建KG数据的逻辑
2. **配置完善**: 添加refinement相关的完整配置参数
3. **测试**: 确保refinement流程在verl中正常工作
4. **集成**: 确保与现有的rollout和reward系统正确集成

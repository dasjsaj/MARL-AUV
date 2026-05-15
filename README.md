# MARL for AUV Swarm Target Tracking

本仓库是面向开源发布整理后的代码包，代码基于 DI-engine 接入多智能体算法，核心环境为六自由度 AUV 集群目标追踪环境，提出方法为 `STG-MAPPO`。该代码所撰写论文已投稿至Science China Information Sciences，在arxiv平台已发布（待更新）。
同时，本项目给出了用于基于MARL的公平性AUV仿真的开源代码，用于研究者快速接入基于AUV的MARL算法对比，并保证了对比的公平性，

> 说明：本 release 目录只保留核心源码、配置、测试脚本、论文绘图脚本和汇总结果。大型 checkpoint、训练日志、半截实验目录、缓存文件没有放入仓库。

## 1. 项目功能

本项目支持：

- 六自由度 AUV 集群目标追踪仿真；
- DI-engine 多智能体训练接口；
- STG-MAPPO 语义增强方法；
- MAPPO、MADDPG、MATD3、HAPPO、MADQN、MASAC 等基线算法实验；
- medium / hard 场景训练；
- 消融实验；
- best checkpoint 鲁棒性压力测试；
- 论文级收敛曲线、柱状图、CSV/XLSX 表格生成。

## 2. 目录结构

```text
OpenMARL_AUV_STG_GitHub/
├── Tracking/
│   ├── auv6dof/                         # AUV 六自由度环境与场景定义
│   │   ├── dynamics.py                   # 六自由度动力学积分与目标运动
│   │   ├── gym_env.py                    # Gym 环境、动作映射、step/reset
│   │   └── scenario_v2.py                # 观测、奖励、语义特征、medium/hard reset
│   ├── di_envs/                          # DI-engine 环境封装
│   │   ├── auv6dof_di_env.py
│   │   └── action_codebook.py
│   ├── configs/                          # 训练与实验配置
│   ├── scripts/                          # 训练、评估、绘图、压力测试脚本
│   ├── tests/                            # 环境与奖励 sanity tests
│   └── marl_orchestrator.py              # 多算法统一训练调度入口
├── artifacts/auv6dof_tmc_2e6/
│   ├── paper_figures/                    # medium 论文图
│   ├── paper_tables/                     # medium 论文表
│   ├── hard_paper_figures/               # hard 论文图
│   ├── hard_paper_tables/                # hard 论文表
│   ├── ablation_medium_4auv/
│   │   ├── ablation_paper_figures/       # 消融图
│   │   └── ablation_paper_tables/        # 消融表
│   └── best_ckpt_stress_table/           # checkpoint 压力测试汇总表
├── requirements.txt
├── .gitignore
└── README.md
```

## 3. 环境安装

推荐使用 Conda：

```powershell
conda create -n auv-stg python=3.9 -y
conda activate auv-stg
python -m pip install -r requirements.txt
```

如果运行时报：

```text
ModuleNotFoundError: No module named 'ding'
```

说明 DI-engine 没有正确安装。可以先尝试：

```powershell
python -m pip install DI-engine
```

同时,如果你需要优化MARL/RL算法或依赖RL/MARL本地库，则在“https://github.com/opendilab/DI-engine”中将DI-engine下在本地，
把 DI-engine 源码放在本仓库同级目录，然后执行（极其建议下在本地，为了避免冗余，本代码没嵌入DI-engine源码）：


```powershell

```

## 4. 快速检查

进入仓库根目录：

```powershell
cd OpenMARL_AUV_STG_GitHub
```

检查 reward：

```powershell
python Tracking\scripts\inspect_reward_terms.py --config Tracking\configs\auv6dof_tmc_medium_2e6.json --steps 300
```

检查动作是否生效：

```powershell
python Tracking\scripts\inspect_action_effect.py --config Tracking\configs\auv6dof_tmc_medium_2e6.json
```

检查观测分布：

```powershell
python Tracking\scripts\inspect_obs_distribution.py --config Tracking\configs\auv6dof_tmc_medium_2e6.json --steps 300
```

随机策略基线：

```powershell
python Tracking\scripts\evaluate_random_policy.py --config Tracking\configs\auv6dof_tmc_medium_2e6.json --episodes 10
```

## 5. 核心环境说明

### 5.1 六自由度动力学

六自由度 AUV 更新在：

```text
Tracking/auv6dof/dynamics.py
```

核心函数：

```text
_integrate_agent_6dof
```

它更新：

- 位置：`x, y, z`
- 姿态：`roll, pitch, yaw`
- 线速度：`u, v, w`
- 角速度：`p, q, r`
- 控制输入：`tau = [Fx, Fy, Fz, K, M, N]`

### 5.2 medium / hard 场景

场景难度在：

```text
Tracking/auv6dof/scenario_v2.py
```

核心函数：

```text
_stage_profile
```

区别：

- `medium`：目标初始距离较近，目标速度较低，初始分布较温和；
- `hard`：目标初始距离更远，目标速度更高，AUV 初始半径更大，追踪更难。

### 5.3 STG-MAPPO 与基线的区别

`STG-MAPPO` 使用：

- semantic observation；
- semantic graph diagnostics；
- semantic reward；
- `velocity3` 低维速度动作抽象。

其他基线默认使用：

- 非语义原始状态；
- `tau6` 六维力/力矩动作；
- 非语义 tracking reward。

这样可以保证论文对比中“只有提出方法使用语义信息”。

## 6. 运行训练

### 6.1 STG-MAPPO medium

单 seed：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase stage1-stg-medium --seeds 0 --random-episodes 1
```

三 seed：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase stage1-stg-medium --seeds 0,1,2 --random-episodes 1
```

### 6.2 medium 主对比

运行 MAPPO、MADDPG、MATD3、HAPPO、MADQN、MASAC：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-medium --algos mappo,maddpg,matd3,happo,madqn,masac --seeds 0,1,2 --random-episodes 1
```

只运行某一个算法，例如 MATD3：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-medium --algos matd3 --seeds 0 --random-episodes 1
```

### 6.3 hard 主对比

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-hard --algos stg_mappo,mappo,maddpg,matd3,happo,madqn,masac --seeds 0,1,2 --random-episodes 1
```

### 6.4 消融实验

只运行 `MAPPO-velocity3-nonsemantic`：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase ablation --algos mappo_velocity3_nonsemantic --seeds 0,1,2 --random-episodes 1
```

只运行 `MAPPO-semantic-state-only`：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase ablation --algos mappo_semantic_state_only --seeds 0,1,2 --random-episodes 1
```

完整 STG-MAPPO 作为主方法，可以直接使用 `stage1-stg-medium` 结果参与消融对比。

## 7. 并行运行建议

可以开多个 PowerShell 窗口，每个窗口跑一个 seed 或一个算法。例如：

窗口 1：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-medium --algos maddpg --seeds 0 --random-episodes 1
```

窗口 2：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-medium --algos matd3 --seeds 0 --random-episodes 1
```

窗口 3：

```powershell
python Tracking\scripts\run_tmc_2e6_suite.py --phase main-medium --algos mappo --seeds 0 --random-episodes 1
```

并行数量建议根据显存和 CPU 调整。若使用 CPU 训练，不建议同时开太多长测。

## 8. 生成论文图和表

medium 图表：

```powershell
python Tracking\scripts\generate_medium_paper_figures_tables.py
```

hard 图表：

```powershell
python Tracking\scripts\generate_hard_paper_figures_tables.py
```

消融图表：

```powershell
python Tracking\scripts\generate_ablation_paper_figures_tables.py
```

checkpoint 压力测试表：

```powershell
python Tracking\scripts\evaluate_best_ckpt_stress_tables.py --scenarios medium,hard --episodes 30 --horizon 500
```

## 9. 主要指标

训练和评估中常用指标如下：

- `eval_return`：评估回报，越高越好；
- `mean_tracking_error`：平均追踪误差，越低越好；
- `tail_mean_target_distance`：评估尾段 AUV 到目标距离，单位 km，越低越好；
- `mean_target_lost`：目标丢失率，越低越好；
- `mean_action_saturation_rate`：动作饱和率，越低越好；
- `mean_control_cost`：控制代价，越低越好；
- `mean_tracking_reward`：追踪奖励分量；
- `mean_semantic_reward`：语义奖励分量，仅 STG-MAPPO 重点分析。

## 10. 输出目录

训练结果默认写入：

```text
artifacts/auv6dof_tmc_2e6/
```

典型目录：

```text
artifacts/auv6dof_tmc_2e6/medium_4auv/auv6dof/<algo>/seed_<seed>/<run_name>/
artifacts/auv6dof_tmc_2e6/hard_4auv/auv6dof/<algo>/seed_<seed>/<run_name>/
```

重要文件：

```text
eval_detail.csv               # 每次 evaluation 的详细指标
summary.json                  # 当前 run 的摘要
exp/result.pkl                # DI-engine 训练步数与结束状态
exp/ckpt/ckpt_best.pth.tar    # best checkpoint
```

注意：这些大型训练输出没有随开源包附带。请自行运行训练生成。

## 11. 常见问题

### 11.1 No module named ding

DI-engine 没装好。执行：

```powershell
python -m pip install DI-engine
```

或安装本地 DI-engine：

```powershell
python -m pip install -e ..\DI-engine --no-deps
```

### 11.2 No module named auv6dof.scenario

当前代码使用：

```text
Tracking/auv6dof/scenario_v2.py
```

请确认运行入口在仓库根目录，且不要手动 import 旧的 `auv6dof.scenario`。

### 11.3 No module named di_envs.tracking_di_env

旧环境入口已经废弃。当前应使用：

```text
Tracking/di_envs/auv6dof_di_env.py
```

### 11.4 eval_detail.csv 行数为什么不同

不同算法在 DI-engine 中的 evaluation 触发频率可能不同。论文图脚本会将 eval event 映射到真实 env step 轴，避免直接用 CSV 行数比较训练步长。

## 12. 注意事项

本 release 包没有包含：

- checkpoint；
- 原始训练日志；
- 临时 Word 草稿；
- 本地 IDE 配置；
- DI-engine 1GB 源码副本；
- 半截实验目录。

如果需要完全复现实验，请根据第 6 节重新运行训练。




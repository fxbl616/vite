# STAR-ART-ViTE v2 结构说明与设计心得

## 1. 它现在还算 STAR 吗？

严格说，**现在的网络主体已经不再是原始 STAR**。

它保留了 STAR 的工程外壳和数据协议，但模型内部的主要交互建模方式已经被 ART 和 ViTE 的思想替换。更准确的叫法应该是：

```text
STAR data/training framework + ART relation graph + ViTE expert routing + motion-prior residual decoder
```

也就是说，它和原始 STAR 的关系主要体现在以下几个方面：

- 仍然使用 STAR 的 `Trajectory_Dataloader`。
- 仍然使用 STAR 的 ETH/UCY leave-one-out 数据组织方式。
- 仍然使用 STAR 的 `nodes_norm` 坐标体系，也就是以观测最后一帧为中心的 normalized trajectory。
- 仍然保留 STAR 原始 dataloader 生成的 `seq_list / nei_list / batch_pednum`。
- 仍然通过 `trainval.py` 和 `processor.py` 训练、测试、保存模型。

但是，原始 STAR 的核心网络逻辑基本已经被替换：

- 原 STAR 的逐帧 autoregressive temporal-spatial transformer 不再作为主体。
- 原 STAR 的两级 `spatial_encoder_1 / spatial_encoder_2` 不再使用。
- 原 STAR 测试时通过多次随机噪声 forward 采样的方式被改成了一次 forward 输出 K 条轨迹。
- 原 STAR 依赖固定邻居 mask 做空间交互，现在改成 learned relation + spatial prior + top-p adaptive pruning。

所以从论文表述上，不建议说“我们只是改进 STAR”。更合理的表述是：

```text
We build upon the STAR codebase and data protocol, but replace its interaction modeling module with an ART- and ViTE-inspired adaptive relational expert architecture.
```

中文就是：

```text
我们基于 STAR 的代码框架和数据协议，重构了其交互建模模块，引入 ART 的时间感知关系图和 ViTE 的专家路由机制。
```

## 2. 原始 STAR 的主要特点

原始 STAR 的思想是：

```text
spatial transformer + temporal transformer + autoregressive prediction
```

它的交互建模主要依赖两点：

1. 每一帧根据 `nei_list` 构造邻接关系。
2. spatial transformer 在这个邻接图上做注意力传播。

然后它逐帧预测未来：

```text
obs -> predict t+1 -> feed back -> predict t+2 -> ...
```

这种方式的优点是：

- 和轨迹时间递推天然匹配。
- 对短期预测比较稳。
- 原代码虽然老，但 inductive bias 很强。

缺点也明显：

- 它没有显式的概率选路机制。
- 它的交互图基本由固定邻居规则控制。
- 多样性主要来自噪声和多次 forward，而不是一次 forward 的多模态 decoder。
- 它没有区分“一阶邻居交互”和“高阶隐式交互”。

这也是为什么一开始你说“STAR 完全没有根据概率选择不同的路的想法”是准确的。

## 3. ART 的思想怎么接进来的？

ART 的核心有两个：

```text
Temporal-Aware Relation Graph (TARG)
Adaptive Interaction Pruning (AIP)
```

在当前代码里，对应位置是：

- `TemporalAwareRelationGraph`
- `AdaptiveInteractionPruning`

### 3.1 Temporal-Aware Relation Graph

原始 STAR 的空间关系更像是：

```text
当前帧谁是谁的邻居
```

而 ART 的思想是：

```text
两个行人在观测历史中的哪些时刻发生了重要交互
```

所以当前模型先把观测 8 帧编码成：

```text
temporal_features: [8, Nv, D]
```

其中：

- `8` 是观测帧数。
- `Nv` 是观测段完整存在的行人数量。
- `D` 默认是 64。

然后对每一对行人 `(i, j)`，在时间维上做 pairwise attention，得到：

```text
relation_features R: [Nv, Nv, D]
relation_weights Wlearned: [Nv, Nv]
```

这比简单地用最后一帧距离或固定邻居 mask 更细，因为它试图学习：

```text
谁影响谁，以及这种影响主要发生在观测历史的哪个时间段。
```

### 3.2 Adaptive Interaction Pruning

ART 的 AIP 不是固定保留 top-k 个邻居，而是用 top-p cumulative probability：

```text
对每个行人 i：
1. 按关系强度排序其他行人。
2. 从强到弱累加关系权重。
3. 保留累计权重达到 p 的最小邻居集合。
```

当前默认：

```text
aip_top_p = 0.75
```

输出：

```text
adjacency A: [Nv, Nv]
```

这个图是稀疏的、行归一化的、每个行人自适应大小的交互图。

## 4. 为什么又加入 STAR 的空间先验？

第一版激进重构效果不好，一个重要原因是：

```text
完全让随机初始化的 attention 去学习交互图太难。
```

行人轨迹预测里，最强的先验其实非常朴素：

```text
距离近的人更可能交互。
原始 STAR 的 nei_list 也包含有效邻居信息。
```

所以 v2 里我加入了空间先验：

```text
Wspatial = distance_prior * neighbor_prior
```

具体来自：

- 最后观测帧的两两距离 `torch.cdist`。
- 原 STAR dataloader 生成的 `nei_list`。

然后和 ART 学到的关系权重融合：

```text
W = (1 - spatial_prior_mix) * Wlearned + spatial_prior_mix * Wspatial
```

默认：

```text
spatial_prior_mix = 0.6
```

也就是说，当前图结构不是纯 ART，也不是纯 STAR，而是：

```text
40% learned temporal relation + 60% spatial/neighbor prior
```

这一步非常关键。它让模型训练初期就有一个合理的交互图，而不是从完全随机的 pairwise attention 开始摸索。

我的判断是：**这是这版 ADE 明显下降的主要原因之一。**

## 5. ViTE 的思想怎么接进来的？

ViTE 的核心是：

```text
one-hop expert + high-order expert + expert router
```

也就是不强迫所有行人都用同一种交互方式，而是让模型按概率选择：

- 这个行人更需要直接邻居交互？
- 还是更需要高阶/群体交互？
- 或者两个都要？

当前模型里对应三个模块：

```text
OneHopExpert
HighOrderExpert
ExpertRouter
```

### 5.1 One-hop Expert

One-hop expert 使用 AIP 剪枝后的邻接矩阵：

```text
A: [Nv, Nv]
```

它做的是直接邻居聚合：

```text
one_hop_nodes = aggregate(pair_nodes over A)
```

输出：

```text
one_hop_nodes: [Nv, D]
```

它适合处理：

- 避让附近行人。
- 跟随近距离同伴。
- 局部密集区域里的交互。

### 5.2 High-order Expert

High-order expert 使用 virtual nodes。

它不是靠堆很多 GNN 层做多跳传播，而是：

```text
real nodes -> virtual nodes -> real nodes
```

直觉上，virtual nodes 像一组可学习的“场景中介节点”，它们先汇聚全局或群体信息，再把这些信息分发回每个行人。

输出：

```text
high_order_nodes: [Nv, D]
```

它适合处理：

- 群体运动趋势。
- 远距离但有共同方向的人。
- 非直接邻居带来的隐式影响。

### 5.3 Expert Router

Expert router 给每个行人输出两个专家的概率：

```text
router_weights: [Nv, 2]
```

其中两个维度分别对应：

```text
[one-hop expert, high-order expert]
```

训练时加入 noisy gating，然后用 top-p 选择专家。最终融合：

```text
routed_nodes =
    router_weight_1 * one_hop_nodes
  + router_weight_2 * high_order_nodes
```

这就是你最开始想要的“根据概率选择不同的路”的核心。

## 6. 为什么加入 motion prior residual decoder？

这是 v2 里我认为最实际、最有效的一步。

第一版 decoder 是：

```text
final_features -> MLP -> future trajectory
```

这让模型从零生成未来 12 帧。问题是：轨迹预测并不是图像生成，短期运动有非常强的物理/运动连续性。一个很强的 baseline 是：

```text
保持最后观测速度不变
```

所以现在 decoder 改成：

```text
baseline(t) = last_pos + t * last_velocity
prediction(t) = baseline(t) + residual(t)
```

其中 residual 由 K 个 MLP head 预测。

这样模型的任务从：

```text
预测完整未来轨迹
```

变成：

```text
预测相对常速度外推的修正量
```

这个任务简单很多。

这也是为什么你看到训练早期 ADE/FDE 下降明显。模型刚开始即使 residual 很小，也能输出一条合理的常速度轨迹，而不是随机轨迹。

## 7. 当前完整数据流

当前模型的数据流可以概括为：

```text
STAR batch
  -> observed trajectory [8, Nv, 2]
  -> position + velocity [8, Nv, 4]
  -> temporal transformer [8, Nv, D]
  -> initial node feature [Nv, D]

temporal features
  -> ART TARG
  -> learned relation Wlearned [Nv, Nv]
  -> relation feature R [Nv, Nv, D]

last observed positions + STAR nei_list
  -> spatial prior Wspatial [Nv, Nv]

Wlearned + Wspatial
  -> AIP top-p pruning
  -> sparse adjacency A [Nv, Nv]

initial node + R + A
  -> relational transformer
  -> pair_nodes [Nv, D]

pair_nodes
  -> one-hop expert [Nv, D]
  -> high-order expert [Nv, D]
  -> expert router [Nv, 2]
  -> routed_nodes [Nv, D]

[initial_nodes, pair_nodes, routed_nodes]
  -> final_features [Nv, 3D]
  -> K residual decoder heads [K, 12, Nv, 2]

last position + last velocity
  -> constant velocity baseline [12, Nv, 2]

baseline + residual
  -> prediction [K, 12, Nv, 2]
  -> scatter back to [K, 12, N, 2]
```

## 8. 当前 loss 和评价指标

训练 loss 是：

```text
best-of-K ADE
+ fde_weight * minFDE
+ moe_weight * router_loss
+ diversity_weight * endpoint_diversity
```

其中：

- `best-of-K ADE` 让 K 条轨迹里最接近 GT 的那条负责主要训练。
- `minFDE` 单独鼓励最终点准确。
- `router_loss` 防止 expert router 坍缩到某一个专家。
- `endpoint_diversity` 防止 K 个 decoder head 全部输出同一条轨迹。

测试指标现在是主流多样本轨迹预测常用的：

```text
minADE_K / minFDE_K
```

当 `sample_num=20` 时，就是：

```text
minADE_20 / minFDE_20
```

这一点和 ART、ViTE、MART、LED、MID 等多样本轨迹预测论文的表格习惯基本一致。

需要注意的是，原始 STAR 的代码风格更老，它通过多次 forward 采样来得到多条轨迹；当前模型是一次 forward 直接输出 K 条轨迹。两者如果比较，必须确认都是同一个 K、同一个数据切分、同一个 minADE/minFDE 定义。

## 9. 我的设计心得

这次改造最重要的经验是：

```text
不能只把新论文模块机械塞进去。
```

第一版激进重构引入了 ART 和 ViTE 的形式，但效果不如原 STAR。原因不是 ART/ViTE 思想没用，而是轨迹预测任务里有几个很强的基础 inductive bias：

1. **短期运动连续性非常强。**
   如果不用常速度或历史速度做先验，decoder 会很难训。

2. **空间距离仍然是最强交互先验之一。**
   完全 learned relation 在小数据集上不一定比手工邻居规则更可靠。

3. **多模态预测需要明确的 K-head 机制。**
   靠噪声多次 forward 可以采样，但不如显式 K decoder heads 直接。

4. **expert routing 需要一个稳定的底图。**
   如果输入图本身很乱，router 选择 one-hop/high-order 也学不好。

所以 v2 的思路是：

```text
先用 STAR/轨迹任务的强先验稳住底座，
再把 ART 和 ViTE 的自适应关系建模放在上面。
```

也就是：

```text
motion prior 保证轨迹不乱
spatial prior 保证交互图不乱
ART 负责学习时间感知关系
ViTE 负责学习不同交互尺度的概率路由
K-head decoder 负责多模态输出
```

这比“纯粹追求新模块复杂度”更有效。

## 10. 当前模型的优点和不足

### 优点

- 比原 STAR 更明确地支持概率选路。
- 比原 STAR 更明确地支持一次 forward 输出 K 条轨迹。
- 引入 ART 的时间感知关系，不只看静态邻居。
- 引入 ViTE 的 one-hop/high-order 专家分工。
- 加入 motion prior 后训练早期明显更稳。
- 空间先验让关系图更符合行人轨迹任务。

### 不足

- 它已经不是原始 STAR 的小改动，而是一个新结构。
- 当前 ART 的 TARG 是简化实现，并非完全复现论文全部细节。
- 当前 ViTE 的 high-order expert 是纯 PyTorch 简化版，没有使用 torch_geometric。
- relation graph 和 expert router 的可解释性还需要可视化验证。
- 目前还没有系统 ablation，不知道各模块分别贡献多少。

## 11. 后续建议做的消融实验

如果要写论文或报告，我建议至少做以下 ablation：

1. 去掉 motion prior：

```text
prediction = residual only
```

2. 去掉 spatial prior：

```text
W = Wlearned only
```

3. 去掉 ART learned relation：

```text
W = Wspatial only
```

4. 只用 one-hop expert。

5. 只用 high-order expert。

6. 固定混合两个 expert，不使用 router。

7. 去掉 diversity loss。

这样可以回答：

```text
到底是 ART 有用，ViTE 有用，还是 motion prior/spatial prior 有用？
```

这对说服别人非常重要。

## 12. 一句话总结

当前模型已经不是原始 STAR 的主体结构，而是一个基于 STAR 数据框架的新混合模型：

```text
ART 负责“谁和谁在什么时候交互”，
ViTE 负责“每个行人应该走一阶交互还是高阶交互路线”，
STAR 保留数据协议和邻居先验，
motion prior 保证轨迹预测符合基本运动规律。
```

我认为真正有效的结合点不是简单地把 ART 和 ViTE 拼起来，而是：

```text
用 STAR 的任务先验稳定图和轨迹，
用 ART/ViTE 增强自适应交互建模。
```


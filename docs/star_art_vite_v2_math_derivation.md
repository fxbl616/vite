# STAR-ART-ViTE v2 数学公式推导

本文档以论文 Method 部分的写法，给出当前 `STAR/src/star.py`、`STAR/src/processor.py` 和 `STAR/src/utils.py` 中 STAR-ART-ViTE v2 的数学描述。重点不是复述原始 STAR、ART 或 ViTE 论文，而是严格解释当前代码实际运行的模型。

## 1. 问题定义

给定一个场景中的行人轨迹片段，模型观察前 $T_o$ 帧，预测未来 $T_p$ 帧。当前实现默认：

$$
T_o=8,\quad T_p=12,\quad K=\texttt{sample\_num}=20.
$$

令第 $i$ 个行人在第 $t$ 帧的二维坐标为：

$$
\mathbf{x}_{t,i} = [x_{t,i}, y_{t,i}]^\top \in \mathbb{R}^2.
$$

STAR 数据预处理会把每个行人的第 $T_o$ 帧坐标作为平移中心，因此模型内部使用的是 normalized coordinate：

$$
\bar{\mathbf{x}}_{t,i}
= \mathbf{x}_{t,i} - \mathbf{x}_{T_o,i}.
$$

模型输入为：

$$
\bar{\mathbf{X}} \in \mathbb{R}^{T \times N \times 2},
$$

其中 $T=19$ 是传入 `STAR.forward` 的长度，$N$ 是当前 batch 中拼接后的行人数。模型只使用前 $T_o=8$ 帧编码，未来 $T_p=12$ 帧只在 loss 和 evaluation 中使用。

最终模型一次 forward 输出 $K$ 条候选未来轨迹：

$$
\hat{\mathbf{Y}}
\in \mathbb{R}^{K \times T_p \times N \times 2}.
$$

其中第 $k$ 条候选、第 $\tau$ 个预测步、第 $i$ 个行人的预测位置为：

$$
\hat{\mathbf{y}}^{(k)}_{\tau,i}
\in \mathbb{R}^2,
\quad
k=1,\dots,K,\quad
\tau=1,\dots,T_p.
$$

## 2. 符号表

| 符号 | 含义 | 默认值或维度 |
|---|---|---|
| $N$ | 当前 batch 中的原始行人数 | 动态变化 |
| $N_v$ | 观测段完整存在的有效行人数 | $N_v \le N$ |
| $T_o$ | 观测长度 | 8 |
| $T_p$ | 预测长度 | 12 |
| $D$ | 模型隐藏维度 `model_dim` | 64 |
| $H$ | attention head 数 `num_heads` | 4 |
| $d_h$ | 每个 head 的维度 | $D/H$ |
| $K$ | 候选轨迹数量 `sample_num` | 20 |
| $M$ | virtual node 数量 `num_virtual_nodes` | 3 |
| $p_a$ | AIP 邻居 top-p 阈值 `aip_top_p` | 0.75 |
| $p_r$ | router top-p 阈值 `router_top_p` | 0.7 |
| $\lambda_s$ | 空间先验混合权重 `spatial_prior_mix` | 0.6 |
| $\sigma_s$ | 距离先验尺度 `spatial_sigma` | 2.0 |
| $\lambda_f$ | FDE loss 权重 `fde_weight` | 0.5 |
| $\lambda_m$ | router loss 权重 `moe_weight` | 0.01 |
| $\lambda_d$ | diversity loss 权重 `diversity_weight` | 0.02 |

## 3. 有效行人选择与输入特征构造

当前模型只对观测段完整存在的行人做编码。令 `seq_list` 给出行人在每一帧是否存在，则有效行人 mask 为：

$$
v_i
= \prod_{t=1}^{T_o} \mathbb{1}[\texttt{seq}_{t,i}=1].
$$

有效行人集合为：

$$
\mathcal{V}=\{i \mid v_i=1\},
\quad
|\mathcal{V}|=N_v.
$$

为了避免不同场景之间发生交互，代码使用 `batch_pednum` 把有效行人切成多个 scene slice。第 $s$ 个场景的有效行人集合记为：

$$
\mathcal{V}_s \subseteq \mathcal{V}.
$$

对每个有效行人，模型使用位置和速度构造 4 维输入特征：

$$
\mathbf{f}_{t,i}
=
[
\bar{x}_{t,i},
\bar{y}_{t,i},
\Delta \bar{x}_{t,i},
\Delta \bar{y}_{t,i}
]^\top
\in \mathbb{R}^4.
$$

速度项定义为：

$$
\Delta \bar{\mathbf{x}}_{t,i}
=
\bar{\mathbf{x}}_{t,i}
-
\bar{\mathbf{x}}_{t-1,i},
\quad t=2,\dots,T_o.
$$

第一帧速度在代码中直接复用第二帧速度：

$$
\Delta \bar{\mathbf{x}}_{1,i}
=
\Delta \bar{\mathbf{x}}_{2,i}.
$$

因此输入特征张量为：

$$
\mathbf{F}
\in
\mathbb{R}^{T_o \times N_v \times 4}.
$$

## 4. 时间编码器

首先用线性层将 4 维位置-速度特征投影到 $D$ 维：

$$
\mathbf{e}_{t,i}
=
\mathbf{W}_{in}\mathbf{f}_{t,i} + \mathbf{b}_{in},
\quad
\mathbf{e}_{t,i}\in\mathbb{R}^{D}.
$$

然后加入时间位置编码：

$$
\tilde{\mathbf{e}}_{t,i}
=
\mathbf{e}_{t,i} + \mathbf{p}_t,
$$

其中 $\mathbf{p}_t$ 是 sinusoidal positional encoding。之后送入一层 temporal Transformer encoder：

$$
\mathbf{Z}
=
\mathrm{Transformer}_{time}(\tilde{\mathbf{E}}),
\quad
\mathbf{Z}\in\mathbb{R}^{T_o \times N_v \times D}.
$$

代码中随后进行 LayerNorm：

$$
\mathbf{Z}
\leftarrow
\mathrm{LN}(\mathbf{Z}).
$$

每个行人的初始节点表征取最后一个观测时刻：

$$
\mathbf{h}^{0}_i
=
\mathbf{Z}_{T_o,i}
\in
\mathbb{R}^{D}.
$$

对应代码变量为 `initial_nodes`，维度是：

$$
\mathbf{H}^{0}\in\mathbb{R}^{N_v\times D}.
$$

## 5. ART 风格时间感知关系图

ART 思想的核心是：行人之间的关系不应只由某一帧距离决定，而应该由一段观测轨迹中的时间交互模式决定。当前实现中的 `TemporalAwareRelationGraph` 对每个场景内的行人对 $(i,j)$ 计算 pairwise temporal attention。

### 5.1 多头时序投影

对时间编码特征 $\mathbf{Z}_{t,i}$ 做 QKV 投影：

$$
[
\mathbf{q}_{t,i},
\mathbf{k}_{t,i},
\mathbf{v}_{t,i}
]
=
\mathbf{W}_{qkv}\mathbf{Z}_{t,i}.
$$

将其拆成 $H$ 个 head。第 $h$ 个 head 的向量为：

$$
\mathbf{q}^{(h)}_{t,i},
\mathbf{k}^{(h)}_{t,i},
\mathbf{v}^{(h)}_{t,i}
\in
\mathbb{R}^{d_h}.
$$

### 5.2 行人对的时间注意力

对同一场景内的目标行人 $i$ 和源行人 $j$，第 $h$ 个 head 在时间维度上计算注意力：

$$
s^{(h)}_{i,j,t}
=
\frac{
(\mathbf{q}^{(h)}_{t,i})^\top
\mathbf{k}^{(h)}_{t,j}
}{
\sqrt{d_h}
}.
$$

对时间维度 softmax：

$$
\alpha^{(h)}_{i,j,t}
=
\frac{
\exp(s^{(h)}_{i,j,t})
}{
\sum_{r=1}^{T_o}\exp(s^{(h)}_{i,j,r})
}.
$$

于是行人 $j$ 对行人 $i$ 的时间关系向量为：

$$
\mathbf{r}^{(h)}_{i,j}
=
\sum_{t=1}^{T_o}
\alpha^{(h)}_{i,j,t}
\mathbf{v}^{(h)}_{t,j}.
$$

拼接所有 head 并线性投影，得到边特征：

$$
\mathbf{r}_{i,j}
=
\mathbf{W}_{o}
\mathrm{Concat}
(
\mathbf{r}^{(1)}_{i,j},
\dots,
\mathbf{r}^{(H)}_{i,j}
)
\in
\mathbb{R}^{D}.
$$

边特征张量维度为：

$$
\mathbf{R}\in\mathbb{R}^{N_v\times N_v\times D}.
$$

### 5.3 关系强度

当前实现同时使用正向边特征 $\mathbf{r}_{i,j}$ 和反向边特征 $\mathbf{r}_{j,i}$ 估计关系强度：

$$
\ell^{art}_{i,j}
=
\mathbf{w}_{e}^{\top}
[
\mathbf{r}_{i,j};
\mathbf{r}_{j,i}
]
b_e.
$$

其中 $[\cdot;\cdot]$ 表示拼接。关系强度经过 sigmoid：

$$
W^{art}_{i,j}
=
\sigma(\ell^{art}_{i,j}).
$$

自环被置零：

$$
W^{art}_{i,i}=0.
$$

跨场景的关系也被置零，因此：

$$
W^{art}_{i,j}=0,
\quad
\text{if } i\in\mathcal{V}_s,\ j\in\mathcal{V}_{s'},\ s\ne s'.
$$

得到的 ART 关系矩阵为：

$$
\mathbf{W}^{art}
\in
\mathbb{R}^{N_v\times N_v}.
$$

## 6. STAR 空间先验混合

只用 learned relation 在小数据集 ETH/UCY 上容易不稳定，因此当前 v2 又加入了 STAR 风格的局部空间先验。这个部分是当前模型仍然保留 STAR 思想的主要位置。

取最后一个观测时刻的位置：

$$
\mathbf{p}_i
=
\bar{\mathbf{x}}_{T_o,i}.
$$

计算行人之间的距离：

$$
d_{i,j}
=
\|\mathbf{p}_i-\mathbf{p}_j\|_2.
$$

距离先验为：

$$
S^{dist}_{i,j}
=
\exp
\left(
-
\frac{d_{i,j}}{\sigma_s}
\right).
$$

同时，STAR 原始 dataloader 已经根据 `neighbor_thred` 生成了最后观测帧的邻居矩阵：

$$
B_{i,j}
=
\texttt{nei\_list}_{T_o,i,j}.
$$

当前代码把二者结合为：

$$
S_{i,j}
=
S^{dist}_{i,j}
\left(
0.25+B_{i,j}
\right).
$$

其中 $0.25$ 表示即使 dataloader 没把两人判定为邻居，距离近的行人也仍然保留一部分弱先验；如果 $B_{i,j}=1$，则该边获得更强的空间权重。随后置零自环：

$$
S_{i,i}=0.
$$

每一行用最大值归一化：

$$
S_{i,j}
\leftarrow
\frac{
S_{i,j}
}{
\max_m S_{i,m}+\epsilon
}.
$$

最终 learned relation 和 spatial prior 线性混合：

$$
\tilde{W}_{i,j}
=
(1-\lambda_s)W^{art}_{i,j}
+\lambda_s S_{i,j}.
$$

当前默认 $\lambda_s=0.6$，这意味着当前图结构更偏向稳定的空间邻近关系，同时仍保留 ART 的时间关系学习能力。

## 7. Adaptive Interaction Pruning

`AdaptiveInteractionPruning` 对每个目标行人的候选邻居做 top-p 累积筛选。它不是固定保留 top-k 个邻居，而是保留累计关系概率达到阈值所需的最小邻居集合。

对某个行人 $i$，令同场景内候选关系强度为：

$$
\tilde{\mathbf{w}}_i
=
[
\tilde{W}_{i,1},
\dots,
\tilde{W}_{i,N_v}
].
$$

先按降序排序，得到排列 $\pi_i$：

$$
\tilde{W}_{i,\pi_i(1)}
\ge
\tilde{W}_{i,\pi_i(2)}
\ge
\dots.
$$

行总强度为：

$$
c_i
=
\sum_j \tilde{W}_{i,j}.
$$

选择集合 $\mathcal{N}_i$ 满足：

$$
\mathcal{N}_i
=
\left\{
\pi_i(r)
\mid
\sum_{q=1}^{r}
\tilde{W}_{i,\pi_i(q)}
\le
p_a c_i
\right\}
\cup
\left\{
\pi_i(r^\star)
\right\},
$$

其中 $r^\star$ 是第一个使累计强度超过 $p_a c_i$ 的位置：

$$
r^\star
=
\min
\left\{
r
\mid
\sum_{q=1}^{r}
\tilde{W}_{i,\pi_i(q)}
>
p_a c_i
\right\}.
$$

这个“额外加入第一个超过阈值的邻居”的设计，可以避免保留集合为空。若某一行本身没有任何有效边，即 $c_i=0$，则该行保持为全零。

剪枝后的邻接矩阵为：

$$
A_{i,j}
=
\begin{cases}
\dfrac{\tilde{W}_{i,j}}
{\sum_{m\in\mathcal{N}_i}\tilde{W}_{i,m}+\epsilon},
& j\in\mathcal{N}_i,\\
0,
& \text{otherwise}.
\end{cases}
$$

因此对每个有有效邻居的行人：

$$
\sum_j A_{i,j}=1.
$$

代码记录的 `aip_density` 为保留边数量占全部可能非自环边数量的比例：

$$
\rho_{aip}
=
\frac{
\sum_{i,j}\mathbb{1}[A_{i,j}>0]
}{
\sum_s |\mathcal{V}_s|(|\mathcal{V}_s|-1)
}.
$$

## 8. 边感知关系 Transformer

剪枝后的邻接矩阵 $\mathbf{A}$ 和边特征 $\mathbf{R}$ 被送入 `RelationalTransformerLayer`。它可以理解为一个只在有效边上做 attention 的 edge-aware Transformer。

对节点特征 $\mathbf{h}_i$ 进行投影：

$$
\mathbf{q}_i=\mathbf{W}_Q\mathbf{h}_i,\quad
\mathbf{k}_j=\mathbf{W}_K\mathbf{h}_j,\quad
\mathbf{v}_j=\mathbf{W}_V\mathbf{h}_j.
$$

边特征也被投影到 key 和 value 空间：

$$
\mathbf{e}^{K}_{i,j}
=
\mathbf{W}^{K}_{e}\mathbf{r}_{i,j},
\quad
\mathbf{e}^{V}_{i,j}
=
\mathbf{W}^{V}_{e}\mathbf{r}_{i,j}.
$$

attention mask 只允许两类连接：

$$
\mathcal{M}_{i,j}
=
\mathbb{1}[A_{i,j}>0]
\lor
\mathbb{1}[i=j].
$$

也就是说，节点可以看见被 AIP 保留的邻居，也可以看见自己。

第 $h$ 个 head 的 attention logit 为：

$$
u^{(h)}_{i,j}
=
\frac{
(\mathbf{q}^{(h)}_i)^\top
(
\mathbf{k}^{(h)}_j
+
\mathbf{e}^{K,(h)}_{i,j}
)
}{
\sqrt{d_h}
}
+
b_{i,j},
$$

其中图权重作为 log bias 注入：

$$
b_{i,j}
=
\begin{cases}
\log(A_{i,j}+\epsilon), & A_{i,j}>0,\\
0, & i=j,\\
-\infty, & \mathcal{M}_{i,j}=0.
\end{cases}
$$

注意力权重为：

$$
\beta^{(h)}_{i,j}
=
\frac{
\exp(u^{(h)}_{i,j})
}{
\sum_{m:\mathcal{M}_{i,m}=1}\exp(u^{(h)}_{i,m})
}.
$$

输出消息为：

$$
\mathbf{o}^{(h)}_i
=
\sum_{j:\mathcal{M}_{i,j}=1}
\beta^{(h)}_{i,j}
(
\mathbf{v}^{(h)}_j
+
\mathbf{e}^{V,(h)}_{i,j}
).
$$

拼接所有 head 后经过输出投影：

$$
\mathbf{o}_i
=
\mathbf{W}_{O}
\mathrm{Concat}
(
\mathbf{o}^{(1)}_i,
\dots,
\mathbf{o}^{(H)}_i
).
$$

再经过残差、LayerNorm 和 FFN：

$$
\tilde{\mathbf{h}}_i
=
\mathrm{LN}
(
\mathbf{h}_i+\mathrm{Dropout}(\mathbf{o}_i)
),
$$

$$
\mathbf{h}^{pair}_i
=
\mathrm{LN}
(
\tilde{\mathbf{h}}_i
+
\mathrm{Dropout}
(
\mathrm{FFN}(\tilde{\mathbf{h}}_i)
)
).
$$

如果有多层关系 Transformer，则递推应用：

$$
\mathbf{H}^{pair}
=
\mathrm{RT}^{L}
(
\mathbf{H}^{0},\mathbf{R},\mathbf{A}
),
$$

当前默认 $L=\texttt{rt\_layers}=1$。

## 9. ViTE 风格双专家交互

ViTE 思想在当前模型中体现为两个不同粒度的交互专家：一个负责一阶邻居交互，一个负责高阶全局交互。router 根据每个节点的状态自适应选择专家。

### 9.1 One-Hop Expert

One-Hop Expert 使用剪枝图 $\mathbf{A}$ 做一阶消息聚合。为了避免完全依赖 AIP 权重，代码又学习一个边门控：

$$
g_{i,j}
=
\sigma
(
\mathrm{MLP}_{gate}
(
[
\mathbf{h}^{pair}_i;
\mathbf{h}^{pair}_j
]
)
).
$$

有效邻接为：

$$
\bar{A}^{1}_{i,j}
=
\frac{
A_{i,j}g_{i,j}
}{
\sum_m A_{i,m}g_{i,m}+\epsilon
}.
$$

一阶邻居消息为：

$$
\mathbf{m}^{1}_i
=
\sum_j
\bar{A}^{1}_{i,j}
\mathbf{W}_{m}
\mathbf{h}^{pair}_j.
$$

最终 one-hop 输出为：

$$
\mathbf{e}^{1}_i
=
\mathrm{LN}
\left(
\mathbf{h}^{pair}_i
+
\mathrm{MLP}_{1}
(
[
\mathbf{h}^{pair}_i;
\mathbf{m}^{1}_i
]
)
\right).
$$

这个专家偏向建模局部直接交互，例如跟随、避让、并行行走等一阶关系。

### 9.2 High-Order Expert

High-Order Expert 使用 $M$ 个可学习 virtual nodes 建模高阶交互。对每个场景单独计算，不跨场景传播。

设第 $\ell$ 个 virtual node 为：

$$
\mathbf{u}_{\ell}\in\mathbb{R}^{D},
\quad
\ell=1,\dots,M.
$$

第一阶段是 real-to-virtual，即 virtual nodes 从真实行人节点收集信息：

$$
a_{\ell,i}
=
\frac{
(\mathbf{W}^{v}_{Q}\mathbf{u}_{\ell})^\top
(\mathbf{W}^{r}_{K}\mathbf{h}^{pair}_i)
}{
\sqrt{D}
}.
$$

对真实节点维度归一化：

$$
\gamma_{\ell,i}
=
\frac{\exp(a_{\ell,i})}
{\sum_{m\in\mathcal{V}_s}\exp(a_{\ell,m})}.
$$

virtual context 为：

$$
\mathbf{c}_{\ell}
=
\sum_{i\in\mathcal{V}_s}
\gamma_{\ell,i}
\mathbf{W}^{r}_{V}
\mathbf{h}^{pair}_i.
$$

更新 virtual node：

$$
\hat{\mathbf{u}}_{\ell}
=
\mathrm{LN}
\left(
\mathbf{u}_{\ell}
+
\mathrm{MLP}_{v}(\mathbf{c}_{\ell})
\right).
$$

第二阶段是 virtual-to-real，即真实节点从更新后的 virtual nodes 读回高阶上下文：

$$
b_{i,\ell}
=
\frac{
(\mathbf{W}^{r}_{Q}\mathbf{h}^{pair}_i)^\top
(\mathbf{W}^{v}_{K}\hat{\mathbf{u}}_{\ell})
}{
\sqrt{D}
}.
$$

对 virtual node 维度归一化：

$$
\eta_{i,\ell}
=
\frac{
\exp(b_{i,\ell})
}{
\sum_{m=1}^{M}\exp(b_{i,m})
}.
$$

真实节点读出的高阶上下文为：

$$
\mathbf{c}^{H}_i
=
\sum_{\ell=1}^{M}
\eta_{i,\ell}
\mathbf{W}^{v}_{V}
\hat{\mathbf{u}}_{\ell}.
$$

最终 high-order 输出为：

$$
\mathbf{e}^{H}_i
=
\mathrm{LN}
\left(
\mathbf{h}^{pair}_i
+
\mathrm{MLP}_{H}
(
\mathbf{c}^{H}_i
)
\right).
$$

这个专家的作用是用少量 virtual nodes 近似高阶群体交互，避免显式构造二跳、三跳图导致复杂度过高。

## 10. Noisy Top-p Expert Router

对每个节点，router 输出两个专家的选择概率。未加噪声的 logit 为：

$$
\mathbf{a}_i
=
\mathbf{W}_{g}\mathbf{h}^{pair}_i
\in
\mathbb{R}^{2}.
$$

训练时加入 noisy gating：

$$
\tilde{\mathbf{a}}_i
=
\mathbf{a}_i
+
\boldsymbol{\epsilon}_i
\odot

\left(
\mathrm{Softplus}(\mathbf{W}_{n}\mathbf{h}^{pair}_i)
+\epsilon_0
\right),
$$

其中：

$$
\boldsymbol{\epsilon}_i\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

测试时不加噪声：

$$
\tilde{\mathbf{a}}_i=\mathbf{a}_i.
$$

专家概率为：

$$
\boldsymbol{\pi}_i
=
\mathrm{Softmax}(\tilde{\mathbf{a}}_i)
=
[
\pi_{i,1},
\pi_{i,2}
].
$$

由于当前只有两个专家，top-p 选择的行为可以直观理解为：

$$
\mathcal{E}_i
=
\begin{cases}
\{\arg\max_e \pi_{i,e}\},
& \max_e\pi_{i,e} \ge p_r,\\
\{1,2\},
& \max_e\pi_{i,e} < p_r.
\end{cases}
$$

也就是说，当 router 很确定时只走一个专家；当两个专家概率接近时，同时使用两个专家。

选中专家后重新归一化：

$$
\omega_{i,e}
=
\frac{
\pi_{i,e}\mathbb{1}[e\in\mathcal{E}_i]
}{
\sum_{m\in\mathcal{E}_i}\pi_{i,m}+\epsilon
}.
$$

融合后的节点表示为：

$$
\mathbf{h}^{route}_i
=
\omega_{i,1}\mathbf{e}^{1}_i
+
\omega_{i,2}\mathbf{e}^{H}_i.
$$

再与关系 Transformer 表示做残差融合：

$$
\mathbf{h}^{R}_i
=
\mathrm{LN}
\left(
\mathbf{h}^{pair}_i+\mathbf{h}^{route}_i
\right).
$$

### Router Balance Loss

为了避免所有节点都选择同一个专家，代码对专家 importance 加入 CV 平方惩罚。专家 $e$ 的 importance 为：

$$
I_e
=
\sum_{i=1}^{N_v}\pi_{i,e}.
$$

router loss 为：

$$
\mathcal{L}_{router}
=
\frac{
\mathrm{Var}(\mathbf{I})
}{
\mathrm{Mean}(\mathbf{I})^2+\epsilon
}.
$$

其中：

$$
\mathbf{I}
=
[I_1,I_2].
$$

这个 loss 会惩罚专家使用极不均衡的情况，从而降低 expert collapse 的风险。

## 11. K 解码头与运动先验残差预测

最终用于解码的特征拼接了三个阶段的节点表示：

$$
\mathbf{z}_i
=
[
\mathbf{h}^{0}_i;
\mathbf{h}^{pair}_i;
\mathbf{h}^{R}_i
]
\in
\mathbb{R}^{3D}.
$$

当前代码使用 $K$ 个独立 decoder head，每个 head 输出一条候选轨迹的残差：

$$
\Delta\hat{\mathbf{Y}}^{(k)}_i
=
\mathrm{Decoder}_{k}(\mathbf{z}_i)
\in
\mathbb{R}^{T_p\times 2}.
$$

写成逐步形式：

$$
\Delta\hat{\mathbf{y}}^{(k)}_{\tau,i}
\in
\mathbb{R}^{2},
\quad
\tau=1,\dots,T_p.
$$

模型不直接预测绝对未来坐标，而是在常速度运动先验上预测残差。最后观测位置为：

$$
\mathbf{p}_i
=
\bar{\mathbf{x}}_{T_o,i}.
$$

最后观测速度为：

$$
\mathbf{v}_i
=
\gamma
(
\bar{\mathbf{x}}_{T_o,i}
-
\bar{\mathbf{x}}_{T_o-1,i}
),
$$

其中 $\gamma$ 是可学习标量，对应代码中的 `velocity_scale`。

第 $k$ 条候选轨迹的第 $\tau$ 步预测为：

$$
\hat{\mathbf{y}}^{(k)}_{\tau,i}
=
\mathbf{p}_i
+
\tau\mathbf{v}_i
+
\Delta\hat{\mathbf{y}}^{(k)}_{\tau,i}.
$$

因此 decoder 的实际任务不是从零生成轨迹，而是修正常速度 baseline。这个设计对 ETH/UCY 这类短时轨迹预测非常关键，因为很多行人的主要运动趋势可以由最近速度解释。

最后，代码把有效行人 $N_v$ 的预测 scatter 回原始 $N$ 个行人位置：

$$
\hat{\mathbf{Y}}
\in
\mathbb{R}^{K\times T_p\times N\times 2}.
$$

无效行人的输出保持为零，但这些行人在 loss mask 中不会参与训练或测试。

## 12. 训练目标

训练时使用 future mask，只监督观测段存在且未来仍存在的行人。第 $\tau$ 个未来步的 mask 为：

$$
m_{\tau,i}
=
v_i
\prod_{r=1}^{\tau}
\mathbb{1}
[
\texttt{seq}_{T_o+r,i}=1
].
$$

预测误差为：

$$
d^{(k)}_{\tau,i}
=
\left\|
\hat{\mathbf{y}}^{(k)}_{\tau,i}
-
\mathbf{y}_{\tau,i}
\right\|_2.
$$

其中 $\mathbf{y}_{\tau,i}$ 是第 $\tau$ 个未来步的 ground truth normalized coordinate。

### 12.1 Best-of-K ADE Loss

每个候选 head 对每个行人的 ADE 为：

$$
\mathrm{ADE}^{(k)}_{i}
=
\frac{
\sum_{\tau=1}^{T_p}m_{\tau,i}d^{(k)}_{\tau,i}
}{
\sum_{\tau=1}^{T_p}m_{\tau,i}+\epsilon
}.
$$

对每个行人选择误差最小的候选：

$$
\mathrm{bestADE}_{i}
=
\min_{k}
\mathrm{ADE}^{(k)}_{i}.
$$

batch 级 best ADE loss 为：

$$
\mathcal{L}_{ADE}
=
\frac{1}{|\mathcal{P}_{valid}|}
\sum_{i\in\mathcal{P}_{valid}}
\mathrm{bestADE}_{i},
$$

其中：

$$
\mathcal{P}_{valid}
=
\left\{
i
\mid
\sum_{\tau=1}^{T_p}m_{\tau,i}>0
\right\}.
$$

### 12.2 Best-of-K FDE Loss

如果行人在最后一个预测步仍然存在，则参与 FDE loss：

$$
\mathcal{P}_{fde}
=
\{i\mid m_{T_p,i}=1\}.
$$

第 $k$ 个候选的 final displacement error 为：

$$
\mathrm{FDE}^{(k)}_{i}
=
d^{(k)}_{T_p,i}.
$$

best FDE 为：

$$
\mathcal{L}_{FDE}
=
\frac{1}{|\mathcal{P}_{fde}|}
\sum_{i\in\mathcal{P}_{fde}}
\min_k
\mathrm{FDE}^{(k)}_{i}.
$$

注意当前实现中的 best FDE 与 best ADE 是独立取最小值：

$$
\min_k \mathrm{FDE}^{(k)}_i
\quad
\text{not necessarily the same } k
\text{ as }
\quad
\arg\min_k \mathrm{ADE}^{(k)}_i.
$$

这与主流 multi-modal trajectory prediction 中常见的 `minADE_K` / `minFDE_K` 评估方式一致。

### 12.3 Trajectory Diversity Loss

为了避免 $K$ 个 decoder head 输出几乎相同的轨迹，代码对最终点加入 diversity regularization。对同一个行人 $i$ 的两个候选 $k$ 和 $l$，最终点距离为：

$$
\delta^{(k,l)}_i
=
\left\|
\hat{\mathbf{y}}^{(k)}_{T_p,i}
-
\hat{\mathbf{y}}^{(l)}_{T_p,i}
\right\|_2.
$$

diversity loss 为：

$$
\mathcal{L}_{div}
=
\frac{1}{|\Omega|}
\sum_{(i,k,l)\in\Omega}
\left[
\max
(
0,
\mu-\delta^{(k,l)}_i
)
\right]^2.
$$

其中 $\mu=\texttt{diversity\_margin}$，默认 $\mu=0.2$，集合 $\Omega$ 包含最后预测步有效的行人和所有 $k\ne l$ 的候选对。

这个 loss 在最小化时会惩罚“终点距离小于 margin”的候选对，因此鼓励不同 decoder head 覆盖不同未来模式。

### 12.4 总训练目标

轨迹 loss 为：

$$
\mathcal{L}_{traj}
=
\mathcal{L}_{ADE}
+
\lambda_f\mathcal{L}_{FDE}.
$$

总 loss 为：

$$
\mathcal{L}
=
\mathcal{L}_{traj}
+
\lambda_m\mathcal{L}_{router}
+
\lambda_d\mathcal{L}_{div}.
$$

对应当前命令中的默认推荐配置：

$$
\lambda_f=0.5,\quad
\lambda_m=0.01,\quad
\lambda_d=0.02.
$$

## 13. 测试指标

测试时不再像原始 STAR 那样循环 forward `sample_num` 次。当前模型一次 forward 直接输出：

$$
\hat{\mathbf{Y}}
\in
\mathbb{R}^{K\times T_p\times N\times 2}.
$$

对于完整存在于预测段的行人：

$$
\mathcal{P}_{full}
=
\left\{
i
\mid
\sum_{\tau=1}^{T_p}m_{\tau,i}=T_p
\right\},
$$

minADE_K 定义为：

$$
\mathrm{minADE}_{K}
=
\frac{1}{|\mathcal{P}_{full}|}
\sum_{i\in\mathcal{P}_{full}}
\min_{k}
\frac{1}{T_p}
\sum_{\tau=1}^{T_p}
d^{(k)}_{\tau,i}.
$$

代码实现中先对时间求和，再在最终除以 $T_p|\mathcal{P}_{full}|$，与上式等价。

minFDE_K 定义为：

$$
\mathrm{minFDE}_{K}
=
\frac{1}{|\mathcal{P}_{full}|}
\sum_{i\in\mathcal{P}_{full}}
\min_{k}
d^{(k)}_{T_p,i}.
$$

这里的 minFDE_K 是独立于 minADE_K 选择最优候选的。也就是说，取得最小 ADE 的候选和取得最小 FDE 的候选可以不是同一条轨迹。这是很多多模态轨迹预测论文默认报告的形式。

## 14. 与 STAR、ART、ViTE 的数学关系

当前网络可以概括为：

$$
\text{STAR-ART-ViTE v2}
=
\text{STAR data protocol}
+
\text{ART relation graph}
+
\text{ViTE expert routing}
+
\text{motion-prior residual decoding}.
$$

更具体地说：

1. STAR 保留的是数据格式、ETH/UCY leave-one-out 流程、normalized coordinate、`seq_list`、`nei_list`、`batch_pednum` 以及空间邻居先验。
2. ART 提供的是时间感知 pairwise relation graph，即 $\mathbf{W}^{art}$ 和 $\mathbf{R}$。
3. ViTE 提供的是 one-hop expert、高阶 virtual-node expert 和 noisy top-p expert router。
4. 当前 v2 额外加入了常速度 motion prior，使 decoder 学习残差而不是直接学习完整未来坐标。

因此，从数学主体看，它已经不是原始 STAR 的固定时空图 Transformer；它更接近一个“时间感知关系图 + 自适应剪枝 + 双专家路由 + 多候选残差解码”的混合模型。

## 15. 前向传播维度总览

| 阶段 | 符号 | 代码变量 | 维度 |
|---|---|---|---|
| 输入 normalized 轨迹 | $\bar{\mathbf{X}}$ | `nodes_norm` | $[19,N,2]$ |
| 有效观测 mask | $\mathbf{v}$ | `valid_mask` | $[N]$ |
| 位置-速度特征 | $\mathbf{F}$ | `observed_features` | $[8,N_v,4]$ |
| 时间编码 | $\mathbf{Z}$ | `temporal_features` | $[8,N_v,D]$ |
| 初始节点 | $\mathbf{H}^{0}$ | `initial_nodes` | $[N_v,D]$ |
| ART 关系强度 | $\mathbf{W}^{art}$ | `relation_weights` before mix | $[N_v,N_v]$ |
| ART 边特征 | $\mathbf{R}$ | `relation_features` | $[N_v,N_v,D]$ |
| 空间先验 | $\mathbf{S}$ | `spatial_prior` | $[N_v,N_v]$ |
| 混合关系 | $\tilde{\mathbf{W}}$ | `relation_weights` after mix | $[N_v,N_v]$ |
| AIP 邻接 | $\mathbf{A}$ | `adjacency` | $[N_v,N_v]$ |
| 关系节点 | $\mathbf{H}^{pair}$ | `pair_nodes` | $[N_v,D]$ |
| one-hop 专家 | $\mathbf{E}^{1}$ | `one_hop_nodes` | $[N_v,D]$ |
| high-order 专家 | $\mathbf{E}^{H}$ | `high_order_nodes` | $[N_v,D]$ |
| router 权重 | $\boldsymbol{\omega}$ | `router_weights` | $[N_v,2]$ |
| 路由节点 | $\mathbf{H}^{R}$ | `routed_nodes` | $[N_v,D]$ |
| 解码特征 | $\mathbf{Z}^{dec}$ | `final_features` | $[N_v,3D]$ |
| K 条候选预测 | $\hat{\mathbf{Y}}$ | `predictions` | $[K,12,N,2]$ |

## 16. 代码对应关系

| 数学模块 | 代码位置 | 主要作用 |
|---|---|---|
| 时间编码 | `STAR.forward`, `input_embedding`, `temporal_encoder` | 将 $[x,y,v_x,v_y]$ 编码为节点时序表示 |
| ART 关系图 | `TemporalAwareRelationGraph` | 生成 $\mathbf{W}^{art}$ 和 $\mathbf{R}$ |
| STAR 空间先验 | `_spatial_relation_prior` | 用距离和 `nei_list` 稳定图结构 |
| AIP 剪枝 | `AdaptiveInteractionPruning` | 从 $\tilde{\mathbf{W}}$ 得到稀疏邻接 $\mathbf{A}$ |
| 边感知交互 | `RelationalTransformerLayer` | 在 $\mathbf{A}$ 和 $\mathbf{R}$ 上做 attention |
| 一阶专家 | `OneHopExpert` | 直接邻居聚合 |
| 高阶专家 | `HighOrderExpert` | virtual nodes 建模全局/高阶交互 |
| 专家路由 | `ExpertRouter` | noisy top-p 选择一个或两个专家 |
| 多头解码 | `TrajectoryDecoder` + `decoder_heads` | 产生 $K$ 条候选残差轨迹 |
| 训练 loss | `maskedBestOfKLoss`, `trajectoryDiversityLoss` | best-of-K + FDE + diversity |
| 测试指标 | `L2forTestK` | 一次 forward 计算 minADE_K / minFDE_K |

## 17. 简短总结

从公式上看，当前模型的核心可以写成：

$$
\hat{\mathbf{Y}}
=
\mathrm{Decode}_{1:K}
\left(
[
\mathbf{H}^{0};
\mathrm{RT}(\mathbf{H}^{0},\mathbf{R},\mathbf{A});
\mathrm{MoE}(\mathrm{RT}(\mathbf{H}^{0},\mathbf{R},\mathbf{A}))
]
\right)
+
\mathrm{CV\ Motion}.
$$

其中：

$$
(\mathbf{W}^{art},\mathbf{R})
=
\mathrm{TARG}(\mathbf{Z}),
$$

$$
\mathbf{A}
=
\mathrm{AIP}
\left(
(1-\lambda_s)\mathbf{W}^{art}
+
\lambda_s\mathbf{S}
\right),
$$

$$
\mathrm{MoE}(\cdot)
=
\omega_1\mathrm{OneHop}(\cdot)
+
\omega_2\mathrm{HighOrder}(\cdot).
$$

这三个公式基本概括了当前 STAR-ART-ViTE v2 的数学骨架：ART 负责“学会谁和谁有关系”，AIP 负责“只保留概率质量最高的交互”，ViTE 路由负责“按节点状态选择直接交互还是高阶交互”，最后用运动先验残差解码生成多模态未来轨迹。

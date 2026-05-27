using Distributions, QuadGK, Statistics
using Base.Threads


"""
    estimate_global_tau_parallel(data, n_iters=20; sample_size=1000, verbose=false)

通过重复抽样估计 BQC 使用的全局 `tau`。

# 参数

- `data`：基因 × 样本表达矩阵。
- `n_iters`：重复抽样次数。
- `sample_size`：每次不放回抽取的基因数量上限。
- `verbose`：是否输出估计过程。

# 返回值

命名元组 `(mean, std, all_values)`。
"""
function estimate_global_tau_parallel(
    data::Matrix{<:Real},
    n_iters::Int = 20;
    sample_size::Int = 1000,
    verbose::Bool = false,
)
    n_genes = size(data, 1)
    n_genes < 2 && error("至少需要 2 个基因才能估计 tau。")
    
    # 1. 计算全局背景 S (所有基因均值的标准差)，这部分全局只需计算一次
    gene_means = mean(data, dims=2)
    S = std(gene_means)
    
    # 2. 预分配数组存储每轮的结果
    tau_results = zeros(Float64, n_iters)
    
    # 确定每轮抽样的基因数量
    actual_sample_size = min(sample_size, n_genes)
    
    verbose && println(">>> 开始并行估计 Tau (线程数: $(nthreads()), 迭代次数: $n_iters)...")

    # 3. 并行循环
    Threads.@threads for k in 1:n_iters
        # 为每轮迭代抽取不重复基因，避免同一基因自配对导致 sigma_D 为 0。
        indices = sample(1:n_genes, actual_sample_size, replace=false)
        
        # 预估对子数量，避免动态扩容
        n_pairs = Int(actual_sample_size * (actual_sample_size - 1) / 2)
        pair_stds = Vector{Float64}(undef, n_pairs)
        
        # 计算随机对子的差值标准差 (sigma_D)
        idx = 1
        for i in 1:actual_sample_size
            for j in i+1:actual_sample_size
                # 直接计算 std(data[i] - data[j])
                # 注意：这里直接操作矩阵切片会有内存分配，在大样本下可以考虑 view 或手写循环进一步优化
                pair_stds[idx] = std(data[indices[i], :] .- data[indices[j], :])
                idx += 1
            end
        end
        
        sigma_D = median(pair_stds)
        
        # 按照公式计算该轮的 Tau
        tau_results[k] = max((sqrt(2) * S) / sigma_D, 2.0)
    end

    # 4. 计算最终统计指标
    mean_tau = mean(tau_results)
    std_tau = std(tau_results)

    if verbose
        println(">>> 估计完成！")
        println("    Mean Tau: $(round(mean_tau, digits=4))")
        println("    Std Dev:  $(round(std_tau, digits=4))")
    end

    return (mean = mean_tau, std = std_tau, all_values = tau_results)
end

"""
    calculate_bayesian_shift_score(k, n, p0, tau)

计算在 BQC 先验下，病例组序关系相对对照组发生稳定切换的后验概率。
"""
function calculate_bayesian_shift_score(k, n, p0, tau)
    # 先验 U 型分布 PDF
    prior(p) = begin
        if p <= 1e-6 || p >= 1-1e-6 return 0.0 end
        z = quantile(Normal(), p)
        return (1/tau) * exp(z^2 * 0.5 * (1 - 1/tau^2))
    end
    
    # 似然函数 (二项分布)
    likelihood(p) = pdf(Binomial(n, p), k)
    
    # 计算后验概率质量
    # 如果 p0 < 0.5 (基准为负), 翻转意味着 p > 0.5
    # 如果 p0 > 0.5 (基准为正), 翻转意味着 p < 0.5 (虽然 REO 训练通常已对齐)
    numerator,   _ = quadgk(p -> likelihood(p) * prior(p), 0.5, 1.0)
    denominator, _ = quadgk(p -> likelihood(p) * prior(p), 0.0, 1.0)
    
    post_prob = numerator / (denominator + 1e-12)
    return p0 < 0.5 ? post_prob : (1.0 - post_prob)
end

"""
    calculate_enhanced_bqc(k1, n1, p0, tau)

计算增强型 BQC 分数。

该分数将后验切换概率转换到负对数空间，并按对照组序关系稳定性加权。
"""
function calculate_enhanced_bqc(k1, n1, p0, tau)
    # 原始贝叶斯得分
    conf = calculate_bayesian_shift_score(k1, n1, p0, tau)
    
    # 1. 计算锚定权重 (惩罚在 Control 中不稳定的对子)
    # p0 越接近 0 或 1，权重越高；越接近 0.5，权重越低
    anchor_weight = abs(p0 - 0.5) * 2.0
    
    # 2. 转换到对数空间以拉开 0.99 之后的差距
    # 防止 conf 恰好等于 1 导致 log(0)
    eps = 1e-15
    nlp_score = -log10(1.0 - conf + eps)
    
    # 3. 综合评分：只有 conf 极高且 p0 极稳的对子才能获得高分
    final_score = nlp_score * anchor_weight
    
    return final_score
end


"""
    generate_bqc_threshold_dict(n0, n1, tau, bqc_limit, p0_threshold; verbose=false)

生成 BQC 临界频数字典。

字典键为对照组中 `g1 > g2` 的次数 `k0`，值为病例组中达到 BQC 阈值所需
的临界次数 `k1`。调用方可据此快速过滤大量候选基因对。
"""
function generate_bqc_threshold_dict(
    n0::Int,
    n1::Int,
    tau::Real,
    bqc_limit::Float64,
    p0_threshold::Float64;
    verbose::Bool = false,
)
    # 结果字典：k0 => (k1_low_bound, k1_high_bound)
    threshold_dict = Dict{Int, Int}()

    verbose && println(">>> 正在生成 BQC 阈值字典 (p0_threshold=$p0_threshold, bqc_limit = $bqc_limit )...")

	# 确定 k0 的有效范围
	k0_low_max  = floor(Int, (0.5 - p0_threshold) * n0) # p0 较小一侧的结束 k0

	for k0 in 0:k0_low_max
        p0 = k0 / n0
        # 寻找右侧显著区间
		# 对应 p0 很大，随着k0递减(p0向中心移动) k1下限越来越小，
		pre_k1 = get(threshold_dict, k0 - 1, nothing)
		fro_k1 = isnothing(pre_k1) ? ceil(Int, n1/2) : pre_k1
		for k1 in fro_k1:n1
            if calculate_enhanced_bqc(k1, n1, p0, tau) >= bqc_limit
                threshold_dict[k0] = k1
                threshold_dict[n0 - k0] = n1 - k1 #对称性
                break # 分数一旦跌落，说明离开显著区
            end
		end

	end
    
    verbose && println("    字典生成完毕：共保留 $(length(threshold_dict)) 种有效稳态情形。")
    return threshold_dict
end

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


#/  """
#/      generate_bqc_threshold_dict(n0, n1, tau, bqc_limit, p0_threshold; verbose=false)
#/  
#/  生成 BQC 临界频数字典。
#/  
#/  字典键为对照组中 `g1 > g2` 的次数 `k0`，值为病例组中达到 BQC 阈值所需
#/  的临界次数 `k1`。调用方可据此快速过滤大量候选基因对。
#/  """
#/  function generate_bqc_threshold_dict(
#/      n0::Int,
#/      n1::Int,
#/      tau::Real,
#/      bqc_limit::Float64,
#/      p0_threshold::Float64;
#/      verbose::Bool = false,
#/  )
#/      # 结果字典：k0 => (k1_low_bound, k1_high_bound)
#/      threshold_dict = Dict{Int, Int}()
#/  
#/      verbose && println(">>> 正在生成 BQC 阈值字典 (p0_threshold=$p0_threshold, bqc_limit = $bqc_limit )...")
#/  
#/  	# 确定 k0 的有效范围
#/  	k0_low_max  = floor(Int, (0.5 - p0_threshold) * n0) # p0 较小一侧的结束 k0
#/  
#/  	for k0 in 0:k0_low_max
#/          p0 = k0 / n0
#/          # 寻找右侧显著区间
#/  		# 对应 p0 很大，随着k0递减(p0向中心移动) k1下限越来越小，
#/  		pre_k1 = get(threshold_dict, k0 - 1, nothing)
#/  		fro_k1 = isnothing(pre_k1) ? ceil(Int, n1/2) : pre_k1
#/  		for k1 in fro_k1:n1
#/              if calculate_enhanced_bqc(k1, n1, p0, tau) >= bqc_limit
#/                  threshold_dict[k0] = k1
#/                  threshold_dict[n0 - k0] = n1 - k1 #对称性
#/                  break # 分数一旦跌落，说明离开显著区
#/              end
#/  		end
#/  
#/  	end
#/      
#/      verbose && println("    字典生成完毕：共保留 $(length(threshold_dict)) 种有效稳态情形。")
#/      return threshold_dict
#/  end
#/  


# ===================================================================
# 1. 计算 REO 频数分布 (多线程并行)
# ===================================================================
function calculate_reo_distribution(data::Matrix{Float64}; verbose = false)
    n_genes, n_samples = size(data)
    n_pairs = div(n_genes * (n_genes - 1), 2)
    
    verbose && println(">>> [1/4] Start to count REOs...")
    verbose && println("    # genes: $n_genes, # samples: $n_samples, # gene pairs: $n_pairs")
    
    # 为每个线程预分配一个统计数组，避免竞争锁
    thread_counts = [zeros(Int, n_samples + 1) for _ in 1:nthreads()]
    
    Threads.@threads for i in 1:(n_genes-1)
        id = threadid()
        @inbounds for j in (i+1):n_genes
            k = 0
            # 高效的样本遍历比较
            for s in 1:n_samples
                k += (data[i, s] > data[j, s])
            end
            # k 的范围是 0 到 n_samples，对应索引 k+1
            thread_counts[id][k + 1] += 1
        end
    end
    
    # 汇总所有线程的统计结果
    total_counts = sum(thread_counts)
    return total_counts
end

# ===================================================================
# 2. 强制对称与经验概率密度 (Empirical PDF) 转换
# ===================================================================
function symmetrize_and_to_pdf(counts::Vector{Int}; verbose = false)
    m = length(counts) - 1
    sym_counts = zeros(Float64, m + 1)
    
    verbose && println(">>> [2/4] Symmetrize and convert frequencies to PDF...")
    
    for k in 0:m
        # 强制 a>b 和 b>a 的随机性对称
        sym_counts[k + 1] = (counts[k + 1] + counts[m - k + 1]) / 2.0
    end
    
    # 转换为离散概率密度：总面积为 1，步长为 dp = 1/m
    dp = 1.0 / m
    emp_pdf = sym_counts ./ (sum(sym_counts) * dp)
    return emp_pdf
end

# ===================================================================
# 3. 分布拟合 (基于内部点的最小二乘法网格搜索)
# ===================================================================
function fit_distributions(emp_pdf::Vector{Float64}; verbose = false)
    m = length(emp_pdf) - 1
    
    # 提取内部点 (规避 p=0 和 p=1 处的无穷大奇异点)
    p_vals = collect(1:m-1) ./ m
    target_pdf = emp_pdf[2:end-1]
    
    verbose && println(">>> [3/4] Fit to the Symmetric Beta distribution ...")
    best_alpha, min_sse_beta = 1.0, Inf
    # alpha 通常在 (0, 1] 之间表示 U型分布
    for alpha in 0.001:0.001:2.0
        pred_pdf = [pdf(Beta(alpha, alpha), p) for p in p_vals]
        sse = sum((pred_pdf .- target_pdf).^2)
        if sse < min_sse_beta
            min_sse_beta = sse
            best_alpha = alpha
        end
    end
    
	verbose && println("          Best-fit α =  $(best_alpha)")
    verbose && println(">>> [3/4] Fit to the Probit-Normal distribution ...")
    best_tau, min_sse_probit = 1.0, Inf
    norm_dist = Normal(0, 1)
    # tau 通常 > 1 表示 U型分布
    for tau in 1.01:0.01:20.0
        pred_pdf = Float64[]
        for p in p_vals
            # 公式: f(p) = (1/tau) * exp( (Phi^-1(p))^2 / 2 * (1 - 1/tau^2) )
            z = quantile(norm_dist, p)
            val = (1.0 / tau) * exp((z^2 / 2.0) * (1.0 - 1.0 / tau^2))
            push!(pred_pdf, val)
        end
        sse = sum((pred_pdf .- target_pdf).^2)
        if sse < min_sse_probit
            min_sse_probit = sse
            best_tau = tau
        end
    end
	verbose && println("          Best-fit τ =  $(best_tau)")
    
    return (alpha=best_alpha, sse_beta=min_sse_beta), (tau=best_tau, sse_probit=min_sse_probit)
end


"""
calculate_enhanced_bqc_hierarchical: 层级贝叶斯增强型 BQC 计算
- k0, n0 : 对照组(control)中 g1 > g2 的样本数和总样本数
- k1, n1 : 病例组(case)中 g1 > g2 的样本数和总样本数
- alpha_global : 通过经验分布拟合出的最优 Symmetric Beta 参数
- n_power : 稳态惩罚幂次（通常设为 1 或 2）
- eps : 数值稳定性微量
"""
function calculate_enhanced_bqc_hierarchical(
    k0::Int, n0::Int, k1::Int, n1::Int, 
    alpha_global::Float64; 
    n_power=1, eps=1e-15
)
    # 1. 计算对照组（健康态）的后验均值，作为更稳健的 p0 锚点
    # Beta 分布后验均值公式: (α + k) / (α + β + n)，由于是对称的，α + β = 2 * alpha_global
    p0_post_mean = (alpha_global + k0) / (2 * alpha_global + n0)
    
    # 2. 构建对照组和病例组的完整后验概率分布
    dist_control = Beta(alpha_global + k0, alpha_global + n0 - k0)
    dist_case    = Beta(alpha_global + k1, alpha_global + n1 - k1)
    
    # 3. 自适应方向的数值积分，计算基础贝叶斯得分 (base BQC)
    steps = 500
    dp = 1.0 / steps
    base_bqc = 0.0
    
    if p0_post_mean >= 0.5
        # 稳态是 g1 > g2，计算 P(θ_case < θ_control)
        for i in 1:steps
            p = (i - 0.5) * dp
            base_bqc += pdf(dist_control, p) * cdf(dist_case, p) * dp
        end
    else
        # 稳态是 g1 < g2，计算 P(θ_case > θ_control)
        for i in 1:steps
            p = (i - 0.5) * dp
            base_bqc += pdf(dist_control, p) * (1.0 - cdf(dist_case, p)) * dp
        end
    end
    
	# base_bqc ∈ [0, 1]
    base_bqc = min(max(base_bqc, 0.0), 1.0 - eps)

	# 4. 对数空间变换, '+eps' to  avoid log(0)
    log_part = -log10(1.0 - base_bqc + eps)
    
    # 5. 稳态锚定加权（使用更加稳健的后验均值）
    weight_part = (abs(p0_post_mean - 0.5) * 2.0)^n_power
    
    # 6. 计算最终的增强型评分
    enhanced_score = log_part * weight_part
    
    return enhanced_score, p0_post_mean
end


"""
generate_bqc_threshold_dict: 
Pre-computes a lookup dictionary mapping control group counts (k0) 
to the minimum required case group counts (k1) to satisfy the BQC threshold.
"""
function generate_bqc_threshold_dict(
    n0::Int,
    n1::Int,
    alpha_global::Float64,
    bqc_limit::Float64,
    p0_threshold::Float64;
    verbose::Bool = false,
)
    # Result dictionary: k0 => k1_minimum_threshold
    threshold_dict = Dict{Int, Int}()

    verbose && println(">>> Generating BQC threshold dictionary (p0_threshold = $p0_threshold, bqc_limit = $bqc_limit)...")

    # Iterate through possible k0 values from 0 up to n0/2
    for k0 in 0:div(n0, 2)
        # Calculate the stable posterior mean for the control group to check p0_threshold
        p0_post_mean = (alpha_global + k0) / (2 * alpha_global + n0)
        
        # Filter based on p0_diff: if it doesn't cross the threshold, we stop searching
        # since abs(p0_post_mean - 0.5) monotonically decreases as k0 approaches n0/2
        if abs(p0_post_mean - 0.5) <= p0_threshold
            break
        end

        # Optimization: the required k1 boundary is monotonic with respect to k0
        pre_k1 = get(threshold_dict, k0 - 1, nothing)
        fro_k1 = isnothing(pre_k1) ? ceil(Int, n1 / 2) : pre_k1
        
        for k1 in fro_k1:n1
            # Call the updated hierarchical BQC function
            score, _ = calculate_enhanced_bqc_hierarchical(
                k0, n0, k1, n1, alpha_global; n_power = 1
            )
            
            if score >= bqc_limit
                threshold_dict[k0] = k1
                threshold_dict[n0 - k0] = n1 - k1 # Enforce mathematical symmetry
                break # Once the score boundary is hit, move to the next k0
            end
        end
    end

    verbose && println("    Dictionary generation complete: retained $(length(threshold_dict)) valid steady-state conditions.")
    return threshold_dict
end


"""
calculate_tdi_metrics: Calculates the Task Difficulty Index (TDI) and classifies the dataset's signaling strength.
- results: The vector of NamedTuples containing all pairs' scores.
- bqc_threshold, p0_threshold: The filtering cutoffs.
"""
function calculate_tdi_metrics(results, bqc_threshold::Float64, p0_threshold::Float64; top_k::Int=50, verbose = false)
    # 1. 提取所有大于 0 的分数用于后续绘制分布图（保留噪声基底）
    plot_scores = [x for x in results if x.score > 0.0]
    
    # 2. 计算满足双重硬门槛的有效特征数 (N_effective)
    n_effective = count(x -> x.score >= bqc_threshold && x.p0_diff > p0_threshold, results)
    
    # 3. 计算排名前 K 的核心特征均值 (mu_top)
    all_scores_sorted = sort([x.score for x in results], rev=true)
    k = min(top_k, length(all_scores_sorted))
    mu_top = k > 0 ? sum(all_scores_sorted[1:k]) / k : 0.0
    
    # 4. 计算 TDI 指标
    tdi_score = log10(n_effective + 1) * mu_top
    
    # 5. 自动打印英文评估报告与风险分级
    verbose && println("\nEvaluate task difficulity metrics ...")
    verbose && println("	Effective Feature Pool (N_effective): $n_effective")
    verbose && println("	Top-$k Core Signal Mean (mu_top): $(round(mu_top, digits=4))")
    verbose && println("	Task Difficulty Index (TDI):      $(round(tdi_score, digits=4))")
    
    #/ if mu_top > 5.0
    #/     verbose && println("-> DIFFICULTY LEVEL: LEVEL 1 (Deterministic / Extremely Easy)")
    #/     verbose && println("-> OVERFITTING RISK: Extremely Low. A few pairs can achieve ~100% accuracy.")
    #/ elseif mu_top >= 2.5
    #/     verbose && println("-> DIFFICULTY LEVEL: LEVEL 2 (Strong Signal / Easy)")
    #/     verbose && println("-> OVERFITTING RISK: Low. Linear classifiers (e.g., Linear SVM) will perform robustly.")
    #/ elseif mu_top >= 1.0
    #/     verbose && println("-> DIFFICULTY LEVEL: LEVEL 3 (Weak Signal / Challenging)")
    #/     verbose && println("-> OVERFITTING RISK: High! Strong regularization (Lasso/Ridge) and Strict Nested CV are required.")
    #/ else
    #/     verbose && println("-> DIFFICULTY LEVEL: LEVEL 4 (Noise Wall / Blind Task)")
    #/     verbose && println("-> OVERFITTING RISK: Critical (~100%). Signal is indistinguishable from random noise.")
    #/ end
    #verbose && println("================================================================\n")
    
    return tdi_score, plot_scores
end

using Random

"""
    BitFeature

多数投票搜索使用的稀疏二值特征表示。
"""
struct BitFeature
    idx::Vector{Int}
    feature_index::Int
end

"""
    build_features(X)

将稠密 `UInt8` 特征矩阵转换为稀疏索引表示。

返回的每个 `BitFeature` 记录一个特征列中取值为 `1` 的样本下标。
"""
function build_features(X::Matrix{UInt8})
    n, m = size(X)
    feats = Vector{BitFeature}(undef, m)

    for j in 1:m
        idx = Int[]
        for i in 1:n
            X[i, j] == 1 && push!(idx, i)
        end
        feats[j] = BitFeature(idx, j)
    end

    return feats
end

"""
    sort_features!(feats, y)

按单特征对正类的支持强度降序排列特征。

返回新的排序结果，不会原地修改输入向量。
"""
function sort_features!(feats::Vector{BitFeature}, y::Vector{UInt8})
    score = zeros(Float64, length(feats))

    @inbounds for j in eachindex(feats)
        s = 0
        for i in feats[j].idx
            s += (y[i] == 1 ? 1 : -1)
        end
        score[j] = s
    end

    return feats[sortperm(score, rev=true)]
end

function add_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] += 1
    end
end

function remove_feature!(counts::Vector{Int16}, feat::BitFeature)
    @inbounds for i in feat.idx
        counts[i] -= 1
    end
end

"""
    mask_to_indices(mask, m)

将位掩码转换为 1-based 特征下标列表。
"""
function mask_to_indices(mask::Union{UInt64, UInt128}, m::Int)
    idx = Int[]
    for j in 1:m
        ((mask >> (j - 1)) & 1 == 1) && push!(idx, j)
    end
    return idx
end

"""
    mask_to_indices(mask, feats)

将位掩码转换为原始特征索引列表。
"""
function mask_to_indices(mask::UInt64, feats::Vector{BitFeature})
    idx = Int[]
    for j in eachindex(feats)
        ((mask >> (j - 1)) & 1 == 1) && push!(idx, feats[j].feature_index)
    end
    return idx
end

"""
    eval_mask_error(mask, X, y; tau=nothing)

计算给定位掩码对应的多数投票分类错误数。
"""
function eval_mask_error(mask::Union{UInt64, UInt128}, X::Matrix{UInt8}, y::Vector{UInt8}; tau::Union{Nothing, Int}=nothing)
    n, m = size(X)
    counts = zeros(Int16, n)
    k = 0

    for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            k += 1
            @inbounds for i in 1:n
                counts[i] += X[i, j]
            end
        end
    end

    threshold = isnothing(tau) ? ((k + 1) >>> 1) : tau
    err = 0
    @inbounds for i in 1:n
        pred = counts[i] >= threshold ? 1 : 0
        err += (pred != y[i])
    end

    return err
end

"""
    permutation_test_pvalue(mask, X, y; B=1000, seed=1, tau=nothing)

对固定特征子集执行标签置换检验，返回观测错误数、p 值和置换错误分布。
"""
function permutation_test_pvalue(
    mask::Union{UInt64, UInt128},
    X::Matrix{UInt8},
    y::Vector{UInt8};
    B::Int = 1000,
    seed::Int = 1,
    tau::Union{Nothing, Int} = nothing,
)
    rng = MersenneTwister(seed)
    obs_err = eval_mask_error(mask, X, y; tau=tau)
    perm_errs = zeros(Int, B)

    @inbounds for b in 1:B
        y_perm = copy(y)
        shuffle!(rng, y_perm)
        perm_errs[b] = eval_mask_error(mask, X, y_perm; tau=tau)
    end

    count = 0
    @inbounds for b in 1:B
        if perm_errs[b] <= obs_err && obs_err > 0
            count += 1
        end
    end

    return (
        observed_error = obs_err,
        p_value = (count + 1) / (B + 1),
        perm_errors = perm_errs,
    )
end

add_bit(mask::UInt128, j::Int) = mask | (UInt128(1) << (j - 1))
remove_bit(mask::UInt128, j::Int) = mask & ~(UInt128(1) << (j - 1))
has_bit(mask::UInt128, j::Int) = (mask >> (j - 1)) & 1 == 1

"""
    compute_error_weighted(counts, k, y; w_pos=1.0, w_neg=1.0)

在给定投票计数下，按类别权重搜索最优阈值并返回加权错误数与阈值。
"""
function compute_error_weighted(
    counts::Vector{Int16},
    k::Int,
    y::Vector{UInt8};
    w_pos::Float64=1.0,
    w_neg::Float64=1.0,
)
    hist_pos = zeros(Int, k + 1)
    hist_neg = zeros(Int, k + 1)

    @inbounds for i in eachindex(y)
        c = counts[i] + 1
        if y[i] == 1
            hist_pos[c] += 1
        else
            hist_neg[c] += 1
        end
    end

    cum_pos = cumsum(hist_pos)
    cum_neg = cumsum(hist_neg)
    total_neg = cum_neg[end]

    best_err = Inf
    best_tau = 0

    @inbounds for tau in 0:k
        fn = tau == 0 ? 0 : cum_pos[tau]
        fp = total_neg - (tau == 0 ? 0 : cum_neg[tau])
        err = w_pos * fn + w_neg * fp

        if err < best_err
            best_err = err
            best_tau = tau
        end
    end

    return best_err, best_tau
end

"""
    compute_class_weights(y)

根据类别频率计算平衡权重。
"""
function compute_class_weights(y::Vector{UInt8})
    n = length(y)
    n_pos = sum(y)
    n_neg = n - n_pos

    w_pos = n / (2 * max(n_pos, 1))
    w_neg = n / (2 * max(n_neg, 1))

    return w_pos, w_neg
end

"""
    eval_mask_error_weighted(mask, X, y; w_pos=1.0, w_neg=1.0)

计算加权错误数，并返回对应的最优阈值。
"""
function eval_mask_error_weighted(
    mask::Union{UInt64, UInt128},
    X::Matrix{UInt8},
    y::Vector{UInt8};
    w_pos::Float64=1.0,
    w_neg::Float64=1.0,
)
    n, m = size(X)

    if mask == 0
        n_pos = sum(y)
        n_neg = n - n_pos
        return min(w_pos * n_pos, w_neg * n_neg), 0
    end

    counts = zeros(Int16, n)
    k = 0

    @inbounds for j in 1:m
        if (mask >> (j - 1)) & 1 == 1
            counts .+= X[:, j]
            k += 1
        end
    end

    return compute_error_weighted(counts, k, y; w_pos=w_pos, w_neg=w_neg)
end

"""
    gray_search_weighted(feats, y)

在特征数不超过 20 时使用 Gray code 穷举搜索最优子集。
"""
function gray_search_weighted(feats::Vector{BitFeature}, y::Vector{UInt8})
    n = length(y)
    m = length(feats)
    counts = zeros(Int16, n)
    w_pos, w_neg = compute_class_weights(y)

    best_err = min(w_pos * sum(y), w_neg * (n - sum(y)))
    best_mask = UInt64(0)
    best_tau = 0
    prev_gray = UInt64(0)
    k = 0
    total = UInt64(1) << m

    @inbounds for t in UInt64(0):(total - 1)
        g = t ⊻ (t >> 1)

        if t > 0
            diff = g ⊻ prev_gray
            j = trailing_zeros(diff) + 1

            if ((g >> (j - 1)) & 1) == 1
                add_feature!(counts, feats[j])
                k += 1
            else
                remove_feature!(counts, feats[j])
                k -= 1
            end
        end

        if k > 0
            err, tau = compute_error_weighted(counts, k, y; w_pos=w_pos, w_neg=w_neg)
            if err < best_err || (err == best_err && k < count_ones(best_mask))
                best_err = err
                best_mask = g
                best_tau = tau
                best_err == 0 && break
            end
        end

        prev_gray = g
    end

    return best_mask, best_err, best_tau
end

"""
    sffs_search_weighted(X, y; max_k=typemax(Int), max_iter=1000)

在特征数较大时使用加权 SFFS 近似搜索最优子集。
"""
function sffs_search_weighted(
    X::Matrix{UInt8},
    y::Vector{UInt8};
    max_k::Int=typemax(Int),
    max_iter::Int=1000,
)
    w_pos, w_neg = compute_class_weights(y)

    function best_single()
        best_j = 1
        best_err = Inf
        best_tau = 0

        for j in 1:size(X, 2)
            mask = UInt128(1) << (j - 1)
            err, tau = eval_mask_error_weighted(mask, X, y; w_pos=w_pos, w_neg=w_neg)

            if err < best_err
                best_err = err
                best_j = j
                best_tau = tau
            end
        end

        return UInt128(1) << (best_j - 1), best_err, best_tau
    end

    current_mask, current_err, current_tau = best_single()
    best_mask = current_mask
    best_err = current_err
    best_tau = current_tau

    iter = 0
    while iter < max_iter
        iter += 1
        improved = false

        best_add_err = current_err
        best_add_j = 0
        for j in 1:size(X, 2)
            if !has_bit(current_mask, j)
                new_mask = add_bit(current_mask, j)
                err, tau = eval_mask_error_weighted(new_mask, X, y; w_pos=w_pos, w_neg=w_neg)

                if err < best_add_err
                    best_add_err = err
                    best_add_j = j
                    best_tau = tau
                end
            end
        end

        if best_add_j == 0
            break
        end

        current_mask = add_bit(current_mask, best_add_j)
        current_err = best_add_err
        improved = true

        while true
            best_remove_err = current_err
            best_remove_j = 0

            for j in 1:size(X, 2)
                if has_bit(current_mask, j)
                    new_mask = remove_bit(current_mask, j)
                    new_mask == 0 && continue

                    err, tau = eval_mask_error_weighted(new_mask, X, y; w_pos=w_pos, w_neg=w_neg)
                    if err < best_remove_err
                        best_remove_err = err
                        best_remove_j = j
                        best_tau = tau
                    end
                end
            end

            if best_remove_j == 0
                break
            end

            current_mask = remove_bit(current_mask, best_remove_j)
            current_err = best_remove_err
            improved = true
        end

        if current_err < best_err ||
           (current_err == best_err && count_ones(current_mask) < count_ones(best_mask))
            best_mask = current_mask
            best_err = current_err
        end

        (!improved || count_ones(current_mask) >= max_k) && break
    end

    return best_mask, best_err, best_tau
end

"""
    select_feature_subset(X, y; max_sffs_k=15, n_permutations=1000, seed=1, verbose=false)

为多数投票模型选择二值特征子集。

当特征数不超过 20 时使用 Gray code 精确搜索；不超过 128 时使用加权 SFFS
近似搜索。返回 `(indices, err, p_value, tau)`。
"""
function select_feature_subset(
    X::Matrix{UInt8},
    y::Vector{UInt8};
    max_sffs_k::Int = 15,
    n_permutations::Int = 1000,
    seed::Int = 1,
    verbose::Bool = false,
)
    all(v -> v == 0 || v == 1, X) || error("X 必须是只包含 0/1 的二值特征矩阵。")
    all(v -> v == 0 || v == 1, y) || error("y 必须是只包含 0/1 的标签向量。")

    _, m = size(X)

    if m <= 20
        verbose && println("Using Gray code for feature subset selection")
        feats = sort_features!(build_features(X), y)
        mask, err, tau = gray_search_weighted(feats, y)
        selected = mask_to_indices(mask, feats)
    elseif m <= 128
        verbose && println("Using SFFS for feature subset selection")
        mask, err, tau = sffs_search_weighted(X, y; max_k=max_sffs_k)
        selected = mask_to_indices(mask, m)
    else
        error("Too many features. Choose another method.")
    end

    isempty(selected) && error("多数投票未选择任何有效特征。")
    mask_for_pvalue = zero(UInt128)
    for idx in selected
        mask_for_pvalue = add_bit(mask_for_pvalue, idx)
    end
    test = permutation_test_pvalue(mask_for_pvalue, X, y; B=n_permutations, seed=seed, tau=tau)

    verbose && println("Permutation Test Result:\t$(test.p_value)")

    return selected, err, test.p_value, tau
end

function _small_reo_data(seed::Int=1)
    return generate_test_data(20, 40; rng=MersenneTwister(seed))
end

function _test_cfg(method::REOMethod)
    return REOConfig(
        method=method,
        low_rank_q=0.0,
        top_diff_n=8,
        bqc_threshold=1.0,
        p0_threshold=0.1,
        max_occurrence=2,
        cor_threshold=0.99,
        target_n=3,
        ss_iterations=5,
        ss_ratio=0.7,
        ss_threshold=0.2,
        verbose=false,
    )
end

@testset "REOB 预测与评估" begin
    data, labels, genes = _small_reo_data(10)
    model = REOModel(REOConfig(method=VotingMethod), [("Gene_1", "Gene_2")], [1.0], 0.0)

    pred = predict_reo(model, data, genes)
    @test pred.preds == (data[1, :] .> data[2, :])
    @test pred.preds == Bool.(labels)

    metrics = evaluate_reo(model, data, genes, labels)
    @test metrics.acc == 1.0
    @test metrics.mcc == 1.0
    @test metrics.auc == 1.0

    wrong_genes = ["Missing_$i" for i in eachindex(genes)]
    @test_throws ErrorException predict_reo(model, data, wrong_genes)
    @test_throws ErrorException evaluate_reo(model, data, genes, ones(Int, length(labels)))
end

@testset "REOB 过滤流程" begin
    data, labels, genes = _small_reo_data(11)
    cfg = _test_cfg(VotingMethod)

    keep_low = REOB.filter_low_rank_genes(data, cfg.low_rank_q)
    @test length(keep_low) == size(data, 1)

    keep_diff = REOB.filter_diff_rank_genes(data, labels, keep_low; top_n=cfg.top_diff_n)
    @test length(keep_diff) <= cfg.top_diff_n
    @test 1 in keep_diff
    @test 2 in keep_diff

    pairs, p_values = REOB.get_top_pairs_parallel_fisher(data, labels, keep_diff; n_top=5)
    @test length(pairs) == length(p_values) == 5
    @test issorted(p_values)

    final_pairs, X = REOB.preprocess_filters(data, labels, genes, cfg)
    @test (1, 2) in final_pairs
    @test size(X, 1) == length(labels)
    @test size(X, 2) == length(final_pairs)

    neutral_confounder = Float64.(repeat([0, 1], length(labels) ÷ 2))
    model_with_confounder = fit_reo(data, labels, genes, cfg; confounders=[neutral_confounder])
    @test model_with_confounder isa REOModel
end

@testset "REOB 测试数据边界" begin
    @test_throws ErrorException generate_test_data(1, 10)
    @test_throws ErrorException generate_test_data(10, 1)
end

@testset "REOB 训练策略" begin
    for method in (VotingMethod, RFMethod, LassoMethod)
        data, labels, genes = _small_reo_data(Int(method) + 20)
        cfg = _test_cfg(method)
        Random.seed!(Int(method) + 100)

        model = fit_reo(data, labels, genes, cfg)
        @test model isa REOModel
        @test model.config.method == method
        @test !isempty(model.final_pairs)
        @test length(model.final_pairs) == length(model.weights)
        @test isapprox(sum(model.weights), 1.0; atol=1e-8)

        metrics = evaluate_reo(model, data, genes, labels)
        @test metrics.acc >= 0.95
        @test metrics.mcc >= 0.9
        @test 0.0 <= metrics.auc <= 1.0
    end
end

@testset "REOB 置换检验" begin
    data, labels, genes = _small_reo_data(30)
    cfg = _test_cfg(VotingMethod)
    Random.seed!(30)
    model = fit_reo(data, labels, genes, cfg)

    result = run_permutation_test(data, labels, genes, model, cfg; n_permutations=2)
    @test 0.0 <= result.p_value <= 1.0
    @test result.observed_mcc >= 0.9
    @test length(result.permuted_mccs) == 2
    @test_throws ErrorException run_permutation_test(data, labels, genes, model, cfg; n_permutations=0)
end

@testset "过滤辅助函数" begin
    data = Float64[
        5 4 1 1
        1 1 5 4
        2 2 2 2
        3 3 3 3
    ]
    labels = [1, 1, 0, 0]
    pairs = [(1, 2), (1, 3), (2, 3), (3, 4)]

    X, aligned_pairs = REOB.build_feature_matrix_aligned(data, pairs, labels)
    @test aligned_pairs == [(1, 2), (1, 3), (3, 2), (3, 4)]
    @test X[:, 1] == Bool[1, 1, 0, 0]
    @test X[:, 3] == Bool[1, 1, 0, 0]

    @test REOB.prune_hub_genes([(1, 2), (1, 3), (2, 3), (4, 5)], 1) == [(1, 2), (4, 5)]

    correlated_X = Bool[
        1 1 0
        1 1 1
        0 0 1
        0 0 0
    ]
    kept_pairs, kept_X = REOB.drop_correlated_features(correlated_X, [(1, 2), (3, 4), (5, 6)], 0.95)
    @test kept_pairs == [(1, 2), (5, 6)]
    @test kept_X == correlated_X[:, [1, 3]]

    bqc_data = Float64[
        1 1 1 1 5 5 5 5
        5 5 5 5 1 1 1 1
        5 5 5 5 1 1 1 1
        1 1 1 1 5 5 5 5
    ]
    bqc_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    filtered = REOB.filter_pairs_with_dict(
        [(1, 2), (3, 4), (1, 4)],
        bqc_data,
        bqc_labels,
        Dict(0 => 3, 4 => 1),
    )
    @test filtered == [(1, 2), (3, 4)]

    reo_vec = vcat(fill(false, 10), fill(true, 10))
    continuous_cf = vcat(collect(1.0:10.0), collect(30.0:39.0))
    categorical_cf = vcat(fill("batch_a", 10), fill("batch_b", 10))
    @test REOB.is_confounded(reo_vec, continuous_cf, 0.05)
    @test REOB.is_confounded(reo_vec, categorical_cf, 0.05)

    data_g, labels_g, genes_g = generate_test_data(10, 20; rng=MersenneTwister(42))
    selected = REOB.filter_genes(data_g, labels_g, genes_g, REOConfig(low_rank_q=0.0, top_diff_n=3))
    @test length(selected) == 3
    @test all(i -> 1 <= i <= size(data_g, 1), selected)
end

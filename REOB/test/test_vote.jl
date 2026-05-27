@testset "多数投票特征搜索" begin
    X = UInt8[
        1 0 0
        1 0 0
        1 0 0
        0 1 1
        0 1 1
        0 1 1
    ]
    y = UInt8[1, 1, 1, 0, 0, 0]

    selected, err, p_value, tau = REOB.select_feature_subset(X, y; n_permutations=20)
    @test selected == [1]
    @test err == 0
    @test 0 < p_value <= 1
    @test tau == 1

    rng = MersenneTwister(7)
    X_sffs = UInt8.(rand(rng, 0:1, 30, 25))
    y_sffs = UInt8.(vcat(ones(Int, 15), zeros(Int, 15)))
    X_sffs[:, 1] .= y_sffs

    selected_sffs, err_sffs, _, tau_sffs = REOB.select_feature_subset(X_sffs, y_sffs; n_permutations=10)
    @test selected_sffs == [1]
    @test err_sffs == 0
    @test tau_sffs == 1

    @test_throws ErrorException REOB.select_feature_subset(reshape(UInt8[2], 1, 1), UInt8[1])
    @test_throws ErrorException REOB.select_feature_subset(reshape(UInt8[1], 1, 1), UInt8[2])
end

@testset "多数投票底层工具" begin
    X = UInt8[
        1 0 0
        1 1 0
        0 1 0
        0 0 0
    ]
    y = UInt8[1, 1, 0, 0]

    feats = REOB.build_features(X)
    @test feats[1].idx == [1, 2]
    @test feats[2].idx == [2, 3]

    sorted = REOB.sort_features!(copy(feats), y)
    @test sorted[1].feature_index == 1
    @test length(sorted) == 3

    @test REOB.mask_to_indices(UInt64(0b101), 3) == [1, 3]
    @test REOB.mask_to_indices(UInt64(0b010), feats) == [2]
    @test REOB.eval_mask_error(UInt64(0b001), X, y) == 0
    @test REOB.eval_mask_error(UInt64(0b010), X, y; tau=1) == 2

    @test REOB.compute_class_weights(UInt8[1, 0, 0, 0]) == (2.0, 2 / 3)
    weighted_err, weighted_tau = REOB.eval_mask_error_weighted(UInt64(0b001), X, y)
    @test weighted_err == 0.0
    @test weighted_tau == 1

    mask = REOB.add_bit(UInt128(0), 2)
    @test REOB.has_bit(mask, 2)
    @test !REOB.has_bit(mask, 1)
    @test REOB.remove_bit(mask, 2) == UInt128(0)

    gray_mask, gray_err, gray_tau = REOB.gray_search_weighted(feats, y)
    gray_indices = REOB.mask_to_indices(UInt64(gray_mask), 3)
    @test !isempty(gray_indices)
    @test all(idx -> idx in (1, 2, 3), gray_indices)
    @test any(idx -> idx in (1, 2), gray_indices)
    @test gray_err == 0.0
    @test gray_tau == 1

    y_sffs = UInt8.(vcat(ones(Int, 15), zeros(Int, 15)))
    X_sffs = zeros(UInt8, 30, 5)
    X_sffs[:, 1] .= y_sffs
    sffs_mask, sffs_err, sffs_tau = REOB.sffs_search_weighted(X_sffs, y_sffs; max_k=3)
    @test REOB.has_bit(sffs_mask, 1)
    @test sffs_err == 0.0
    @test sffs_tau == 1
end

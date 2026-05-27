@testset "BQC 统计工具" begin
    data = Float64[
        1 2 3 4 5 6
        2 3 4 5 6 7
        6 5 4 3 2 1
    ]

    tau = REOB.estimate_global_tau_parallel(data, 3; sample_size=3)
    @test tau.mean >= 2.0
    @test tau.std >= 0.0
    @test length(tau.all_values) == 3
    @test all(>=(2.0), tau.all_values)
    @test_throws ErrorException REOB.estimate_global_tau_parallel(reshape([1.0, 2.0], 1, 2))

    right_shift = REOB.calculate_bayesian_shift_score(10, 10, 0.1, 2.0)
    left_shift = REOB.calculate_bayesian_shift_score(0, 10, 0.9, 2.0)
    @test right_shift > 0.99
    @test left_shift > 0.99

    strong_score = REOB.calculate_enhanced_bqc(10, 10, 0.1, 2.0)
    unstable_score = REOB.calculate_enhanced_bqc(5, 10, 0.5, 2.0)
    @test strong_score > unstable_score
    @test unstable_score == 0.0

    threshold_dict = REOB.generate_bqc_threshold_dict(10, 10, 2.0, 1.0, 0.2)
    @test !isempty(threshold_dict)
    for (k0, k1) in threshold_dict
        @test haskey(threshold_dict, 10 - k0)
        @test threshold_dict[10 - k0] == 10 - k1
    end
end

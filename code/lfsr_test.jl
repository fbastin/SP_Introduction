# Linear Feedback Shift Register (LFSR) / Tausworthe generator

struct LFSR
    k::Int      # register length / degree
    r::Int      # tap position (x_{n-r}), 1 <= r < k
    w::Int      # number of bits read for each output
    nu::Int     # spacing between consecutive bit reads
    mask_k::UInt64
    mask_r::UInt64
end

function LFSR(k::Int, r::Int, w::Int, nu::Int)
    @assert k <= 64 "k must be <= 64"
    mask_k = UInt64(1) << (k - 1)
    mask_r = UInt64(1) << (r - 1)
    LFSR(k, r, w, nu, mask_k, mask_r)
end

mutable struct LFSRState
    state::UInt64
end

function step!(s::LFSRState, lfsr::LFSR)
    feed = ((s.state & lfsr.mask_r) == 0 ? UInt64(0) : UInt64(1)) ⊻ 
           ((s.state & lfsr.mask_k) == 0 ? UInt64(0) : UInt64(1))
    
    s.state = ((s.state << 1) | feed) & ((UInt64(1) << lfsr.k) - 1)
    return feed
end

function next_u!(s::LFSRState, lfsr::LFSR)
    acc = UInt64(0)
    for l in 0:(lfsr.w - 1)
        bit = UInt64(0)
        for _ in 1:lfsr.nu
            bit = s.state & 1
            step!(s, lfsr)
        end
        acc = (acc << 1) | bit
    end
    return Float64(acc) / Float64(UInt64(1) << lfsr.w)
end

function period(lfsr::LFSR, seed::UInt64)
    s = LFSRState(seed)
    n = 1
    step!(s, lfsr)
    while s.state != seed && n < (1 << lfsr.k)
        step!(s, lfsr)
        n += 1
    end
    return n
end

function output_period(lfsr::LFSR, seed::UInt64)
    s = LFSRState(seed)
    # copy state for u0
    # fixed
    u0 = next_u!(s, lfsr)
    
    n = 1
    while n < (1 << lfsr.k)
        u = next_u!(s, lfsr)
        u == u0 && return n
        n += 1
    end
    return -1
end

lfsr = LFSR(9, 5, 16, 1)
s = LFSRState(UInt64(1))

println("First 10 outputs:")
for _ in 1:10
    println(next_u!(s, lfsr))
end

println("\nPeriod of the bit sequence = ", period(lfsr, UInt64(1)),
        " (expected 2^9-1 = ", (1 << 9) - 1, ")")

println("\nEffect of nu on the OUTPUT period (2^9-1 = 511):")
for nu in 1:7
    l = LFSR(9, 5, 16, nu)
    p = output_period(l, UInt64(1))
    println("  nu = $nu -> output period = $p (expected ",
            p < 0 ? "n/a" : string(511 ÷ gcd(nu, 511)), ")")
end

s = LFSRState(UInt64(1))
n = 100_000
mean_val = sum(next_u!(s, lfsr) for _ in 1:n) / n
println("\nMean of $n draws: ", mean_val, " (expect ~0.5)")

mutable struct Taus88
    s1::UInt32
    s2::UInt32
    s3::UInt32
end

function next_u!(rng::Taus88)
    b1 = (((rng.s1 << 13) ⊻ rng.s1) >> 19)
    rng.s1 = (((rng.s1 & 0xFFFFFFFE) << 12) ⊻ b1)
    
    b2 = (((rng.s2 << 2)  ⊻ rng.s2) >> 25)
    rng.s2 = (((rng.s2 & 0xFFFFFFF8) << 4)  ⊻ b2)
    
    b3 = (((rng.s3 << 3)  ⊻ rng.s3) >> 11)
    rng.s3 = (((rng.s3 & 0xFFFFFFF0) << 17) ⊻ b3)
    
    return Float64(rng.s1 ⊻ rng.s2 ⊻ rng.s3) / 4294967296.0
end

println("\n--- Taus88 ---")
taus = Taus88(12345, 67890, 13579)
mean_taus = sum(next_u!(taus) for _ in 1:n) / n
println("Mean of $n draws (Taus88): ", mean_taus, " (expect ~0.5)")


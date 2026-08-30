import Random

const M0 = UInt32(0xD2511F53)
const M1 = UInt32(0xCD9E8D57)
const W0 = UInt32(0x9E3779B9)
const W1 = UInt32(0xBB67AE85)
const ROUNDS = 10

function philox_round(x::NTuple{4,UInt32}, k::NTuple{2,UInt32})
    m0 = widemul(x[1], M0)
    m1 = widemul(x[3], M1)
    hi0, lo0 = (m0 >>> 32) % UInt32, m0 % UInt32
    hi1, lo1 = (m1 >>> 32) % UInt32, m1 % UInt32
    y = (hi1 ⊻ x[2] ⊻ k[1], lo1, hi0 ⊻ x[4] ⊻ k[2], lo0)
    return y, (k[1] + W0, k[2] + W1)
end

function philox(C::NTuple{4,UInt32}, K::NTuple{2,UInt32})
    x, k = C, K
    for _ in 1:ROUNDS
        x, k = philox_round(x, k)
    end
    return x
end

mutable struct PhiloxRNG <: Random.AbstractRNG
    ctr::UInt128
    key::NTuple{2,UInt32}
    buffer::NTuple{4,UInt32}
    idx::Int
end

function PhiloxRNG(seed_key::NTuple{2,UInt32} = (UInt32(0), UInt32(0)), start_ctr::UInt128 = UInt128(0))
    PhiloxRNG(start_ctr, seed_key, (UInt32(0), UInt32(0), UInt32(0), UInt32(0)), 5)
end

function Random.rand(rng::PhiloxRNG, ::Type{UInt32})
    if rng.idx > 4
        c = rng.ctr
        c_tuple = (
            (c & 0xFFFFFFFF) % UInt32,
            ((c >> 32) & 0xFFFFFFFF) % UInt32,
            ((c >> 64) & 0xFFFFFFFF) % UInt32,
            ((c >> 96) & 0xFFFFFFFF) % UInt32
        )
        rng.buffer = philox(c_tuple, rng.key)
        rng.ctr += 1
        rng.idx = 1
    end
    
    val = rng.buffer[rng.idx]
    rng.idx += 1
    return val
end

function Random.rand(rng::PhiloxRNG, ::Type{UInt64})
    lo = Random.rand(rng, UInt32)
    hi = Random.rand(rng, UInt32)
    return (UInt64(hi) << 32) | UInt64(lo)
end

Random.rng_native_52(::PhiloxRNG) = UInt64

rng = PhiloxRNG()
println(rand(rng))
println(rand(rng, Float64))
println(rand(rng, 5))
println(rand(rng, 1:100))

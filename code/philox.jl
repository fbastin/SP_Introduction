# Philox4x32-10 counter-based generator (Salmon, Moraes, Dror & Shaw, SC 2011)
# Optimized implementation for native performance (Julia Random API)

import Random
import RandomDataStreams

const M0 = UInt32(0xD2511F53)   # Philox multipliers
const M1 = UInt32(0xCD9E8D57)
const W0 = UInt32(0x9E3779B9)   # odd part of phi * 2^32 (Weyl sequence step)
const W1 = UInt32(0xBB67AE85)
const ROUNDS = 10               # Philox4x32-10

# one Philox round: bijection on (x0,x1,x2,x3) with the current subkeys (k0,k1)
@inline function philox_round(x::NTuple{4,UInt32}, k::NTuple{2,UInt32})
    m0 = widemul(x[1], M0)
    m1 = widemul(x[3], M1)
    hi0, lo0 = (m0 >>> 32) % UInt32, m0 % UInt32
    hi1, lo1 = (m1 >>> 32) % UInt32, m1 % UInt32
    y = (hi1 ⊻ x[2] ⊻ k[1], lo1, hi0 ⊻ x[4] ⊻ k[2], lo0)
    return y, (k[1] + W0, k[2] + W1)
end

# Philox4x32-10: R = E_K(C), one 128-bit block of random bits
function philox(C::NTuple{4,UInt32}, K::NTuple{2,UInt32})
    x, k = C, K
    for _ in 1:ROUNDS
        x, k = philox_round(x, k)
    end
    return x
end

# Counter-based stream integrated into the Julia ecosystem (AbstractRNG).
# Optimized state:
# - ctr is a native UInt128 (incrementing is extremely fast without carry loops)
# - buffer stores the 4 generated values to avoid wasting bits.
mutable struct PhiloxRNG <: RandomDataStreams.AbstractStreamableRNG
    ctr::UInt128
    key::NTuple{2,UInt32}
    buffer::NTuple{4,UInt32}
    idx::Int
end

# Default constructor
function PhiloxRNG(seed_key::NTuple{2,UInt32} = (UInt32(0), UInt32(0)), start_ctr::UInt128 = UInt128(0))
    # idx = 5 forces the generation of a block on the very first call to rand()
    PhiloxRNG(start_ctr, seed_key, (UInt32(0), UInt32(0), UInt32(0), UInt32(0)), 5)
end

# Extracts the 4 UInt32 from the 128-bit counter
@inline function _ctr_to_tuple(c::UInt128)
    return (
        (c & 0xFFFFFFFF) % UInt32,
        ((c >> 32) & 0xFFFFFFFF) % UInt32,
        ((c >> 64) & 0xFFFFFFFF) % UInt32,
        ((c >> 96) & 0xFFFFFFFF) % UInt32
    )
end

# Random Interface: Generation of a native UInt32
function Random.rand(rng::PhiloxRNG, ::Type{UInt32})
    if rng.idx > 4
        rng.buffer = philox(_ctr_to_tuple(rng.ctr), rng.key)
        rng.ctr += 1
        rng.idx = 1
    end
    
    val = rng.buffer[rng.idx]
    rng.idx += 1
    return val
end

# Random Interface: Generation of a UInt64 (combines 2 UInt32 from the buffer)
function Random.rand(rng::PhiloxRNG, ::Type{UInt64})
    lo = Random.rand(rng, UInt32)
    hi = Random.rand(rng, UInt32)
    return (UInt64(hi) << 32) | UInt64(lo)
end

# Signals to Julia that this generator can directly provide 52 high-quality bits
# (necessary for quickly generating Float64).
Random.rng_native_52(::PhiloxRNG) = UInt64


# =========================================================================
# Demonstration of CBRNG properties (Jump ahead, reproducibility, etc.)
# =========================================================================

hex4(u::UInt32) = string(u, base = 16, pad = 8)

println("Block E_K(0) with K = (0,0) [hex]:")
block = philox((UInt32(0), UInt32(0), UInt32(0), UInt32(0)),
               (UInt32(0), UInt32(0)))
println("  ", join(map(hex4, block), " "), "\n")


println("Blocks at counters 0, 1, ..., 4 [hex]:")
for i in 0:4
    blk = philox(_ctr_to_tuple(UInt128(i)), (UInt32(0), UInt32(0)))
    println("  C = $i -> ", join(map(hex4, blk), " "))
end

# same (counter, key) always yields identical bits: reproducibility
println("\nReproducibility:")
println("  philox(0, K=0) again  -> ", join(map(hex4,
        philox((UInt32(0), UInt32(0), UInt32(0), UInt32(0)),
               (UInt32(0), UInt32(0)))), " "))

# different key => independent stream (jump to a different stream)
println("\nSame counter, different key:")
println("  philox(0, K=(1,0))  -> ", join(map(hex4,
        philox((UInt32(0), UInt32(0), UInt32(0), UInt32(0)),
               (UInt32(1), UInt32(0)))), " "))

# moving the counter forward is the whole "transition": E_K(C + 10^6)
# Very simple with a UInt128!
println("\nJump ahead: E_K(C + 10^6) achieved by incrementing the counter only:")
C_128 = UInt128(1_000_000)
println("  E_K(10^6) = ", join(map(hex4, philox(_ctr_to_tuple(C_128), (UInt32(0), UInt32(0)))), " "), "\n")

# moment / uniformity check on 10^5 (uniform) draws using Standard Julia Random API
rng = PhiloxRNG((UInt32(7), UInt32(3)))
n = 100_000
# We use rand(rng) directly: it returns a Float64 in [0, 1)!
s = sum(rand(rng) for _ in 1:n) / n
println("Mean of $n draws via rand(rng): ", s, "   (expect ~0.5)")

# reference vectors from the Random123 C implementation (v[0] = least significant)
refs = [
    ((UInt32(0), UInt32(0), UInt32(0), UInt32(0)), (UInt32(0), UInt32(0))),
    ((UInt32(1), UInt32(0), UInt32(0), UInt32(0)), (UInt32(0), UInt32(0))),
    ((UInt32(2), UInt32(0), UInt32(0), UInt32(0)), (UInt32(0), UInt32(0))),
    ((UInt32(0), UInt32(0), UInt32(0), UInt32(0)), (UInt32(1), UInt32(0))),
]
expected = [
    (UInt32(0x6627e8d5), UInt32(0xe169c58d), UInt32(0xbc57ac4c), UInt32(0x9b00dbd8)),
    (UInt32(0xf8e4cca4), UInt32(0x5cb200db), UInt32(0xb1a574eb), UInt32(0x097eff67)),
    (UInt32(0x04faa329), UInt32(0x51c732a6), UInt32(0x241513ad), UInt32(0x459135e4)),
    (UInt32(0xe3e80670), UInt32(0xe50a0ebc), UInt32(0x95f222c0), UInt32(0xb615aa27)),
]
println("\nMatches random123 reference vectors: ",
        all(philox(c, k) == e for ((c, k), e) in zip(refs, expected)))

println("\n--- Examples of integration with Julia ---")
rng2 = PhiloxRNG()
println("rand(rng2)             = ", rand(rng2))
println("rand(rng2, 5)          = ", rand(rng2, 5))
println("rand(rng2, 1:100, 3)   = ", rand(rng2, 1:100, 3))

# =========================================================================
# RandomDataStreams.jl API integration
# =========================================================================

mutable struct PhiloxGen <: RandomDataStreams.AbstractRNGStream
    next_key_hi::UInt32
    next_key_lo::UInt32
end
PhiloxGen() = PhiloxGen(0, 0)

# Generate the next independent stream (new key)
function RandomDataStreams.next_stream!(gen::PhiloxGen)
    key = (gen.next_key_hi, gen.next_key_lo)
    if gen.next_key_lo == typemax(UInt32)
        gen.next_key_lo = 0
        gen.next_key_hi += 1
    else
        gen.next_key_lo += 1
    end
    return PhiloxRNG(key, UInt128(0))
end

# In Philox, a substream can be defined by a large jump in the counter, e.g., 2^64
function RandomDataStreams.next_substream!(rng::PhiloxRNG)
    sub_idx = (rng.ctr >> 64) + 1
    rng.ctr = UInt128(sub_idx) << 64
    rng.idx = 5 # force buffer flush
    return rng
end

function RandomDataStreams.reset_substream!(rng::PhiloxRNG)
    sub_idx = (rng.ctr >> 64)
    rng.ctr = UInt128(sub_idx) << 64
    rng.idx = 5
    return rng
end

function RandomDataStreams.reset_stream!(rng::PhiloxRNG)
    rng.ctr = 0
    rng.idx = 5
    return rng
end

function RandomDataStreams.get_state(rng::PhiloxRNG)
    return (rng.ctr, rng.key, rng.buffer, rng.idx)
end

# advance_state!(rng, e, c) jumps by 2^e + c if e > 0, -2^(-e) + c if e < 0, or c if e == 0
function RandomDataStreams.advance_state!(rng::PhiloxRNG, e::Integer, c::Integer)
    n = if e == 0
        BigInt(c)
    elseif e > 0
        BigInt(2)^BigInt(e) + BigInt(c)
    else
        -BigInt(2)^BigInt(-e) + BigInt(c)
    end
    
    # Counter arithmetic modulo 2^128
    mask128 = (BigInt(1) << 128) - 1
    rng.ctr = UInt128((BigInt(rng.ctr) + n) & mask128)
    rng.idx = 5
    return rng
end

# Philox4x32-10 counter-based generator (Salmon, Moraes, Dror & Shaw, SC 2011)
# Following the slides "Random Numbers":
#   R = E_K(C), a stateless bijective map from a counter C = (c_1,...,c_N)
#   with N = 4 32-bit words and a key K = (k_0,k_1); M = N/2 subkeys.
#   The sequence follows by incrementing the counter: C, C+1, C+2, ...

const M0 = UInt32(0xD2511F53)   # Philox multipliers
const M1 = UInt32(0xCD9E8D57)
const W0 = UInt32(0x9E3779B9)   # odd part of phi * 2^32 (Weyl sequence step)
const W1 = UInt32(0xBB67AE85)
const ROUNDS = 10               # Philox4x32-10

# one Philox round: bijection on (x0,x1,x2,x3) with the current subkeys (k0,k1)
#   y0 = hi1 xor x1 xor k0      y2 = hi0 xor x3 xor k1
#   y1 = lo1                    y3 = lo0
# where (hi_j, lo_j) are the high/low 32-bit halves of M_j * x_j.
# The subkeys advance by the Weyl sequence between rounds: (k0,k1) += (W0,W1).
function philox_round(x::NTuple{4,UInt32}, k::NTuple{2,UInt32})
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

# counter-based stream: the state is just the counter (no internal transition)
mutable struct PhiloxRNG
    ctr::Vector{UInt32}          # position in the stream
    key::NTuple{2,UInt32}        # identifies the stream
end

# incrementing the counter = jumping ahead by one block
function inc!(c::AbstractVector{UInt32})
    for i in eachindex(c)
        c[i] += UInt32(1)
        c[i] != 0 && return c
    end
    return c
end

# next block of four 32-bit uniforms, then move the counter forward
function next_block!(rng::PhiloxRNG)
    block = philox(Tuple(rng.ctr), rng.key)
    inc!(rng.ctr)
    return block
end

hex4(u::UInt32) = string(u, base = 16, pad = 8)
u01(x::UInt32) = x / 2.0^32     # output function: map to (0,1)

println("Block E_K(0) with K = (0,0) [hex]:")
block = philox((UInt32(0), UInt32(0), UInt32(0), UInt32(0)),
               (UInt32(0), UInt32(0)))
println("  ", join(map(hex4, block), " "), "\n")

rng = PhiloxRNG(UInt32[0, 0, 0, 0], (UInt32(0), UInt32(0)))

# C, C+1, C+2, ... yields successive blocks
println("Blocks at counters 0, 1, ..., 4 [hex]:")
for i in 0:4
    println("  C = $i -> ", join(map(hex4, next_block!(rng)), " "))
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
println("\nJump ahead: E_K(C + 10^6) achieved by incrementing the counter only:")
C = Vector{UInt32}(UInt32[0, 0, 0, 0])
for _ in 1:1_000_000
    inc!(C)
end
println("  E_K(10^6) = ", join(map(hex4, philox(Tuple(C), (UInt32(0), UInt32(0)))), " "), "\n")

# moment / uniformity check on 10^5 (uniform) draws
rng = PhiloxRNG(UInt32[0, 0, 0, 0], (UInt32(7), UInt32(3)))
n = 100_000
s = sum((blk = next_block!(rng);
         u01(blk[1]) + u01(blk[2]) + u01(blk[3]) + u01(blk[4])) for _ in 1:n) / (4n)
println("Mean of $n draws: ", s, "   (expect ~0.5)")

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
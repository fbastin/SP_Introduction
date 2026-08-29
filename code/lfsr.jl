# Linear Feedback Shift Register (LFSR) / Tausworthe generator
# Following the slides "Random Numbers":
#   x_n = (x_{n-r} + x_{n-k}) mod 2,  with addition mod 2 = xor
# and the output accumulated over w bits:
#   u_n = sum_{l=1}^{w} x_{n*nu+l-1} 2^{-l} = .x_{n*nu} x_{n*nu+1} ... x_{n*nu+w-1}

# The register stores reg[j] = x_{n-j} (reg[1] = most recent bit).
# The recurrence is a linear recurrence over F_2, with characteristic
# polynomial z^k + z^(k-r) + 1, which must be primitive to reach the
# maximum period 2^k - 1 (here: z^9 + z^4 + 1, i.e. r = k - 4 = 5).
struct LFSR
    k::Int      # register length / degree
    r::Int      # tap position (x_{n-r}), 1 <= r < k
    w::Int      # number of bits read for each output
    nu::Int     # spacing between consecutive bit reads
end

# step the register once: feed x_n = x_{n-r} xor x_{n-k}, then shift
function step!(reg::Vector{UInt8}, lfsr::LFSR)
    feed = reg[lfsr.r] ⊻ reg[lfsr.k]
    for j in (lfsr.k - 1):-1:1
        reg[j + 1] = reg[j]
    end
    reg[1] = feed
    return reg
end

# produce one number in (0,1) by reading w bits spaced by nu
function next_u!(reg::Vector{UInt8}, lfsr::LFSR)
    u = 0.0
    for l in 0:(lfsr.w - 1)
        bit = 0.0
        for _ in 1:lfsr.nu
            bit = Float64(reg[1])   # read current bit x_{n*nu + l}
            step!(reg, lfsr)
        end
        u += bit * 2.0^(-(l + 1))
    end
    return u
end

# number of states visited before returning to seed (rho = 2^k - 1 for max period)
function period(lfsr::LFSR, seed::Vector{UInt8})
    reg = copy(seed)
    n = 1
    step!(reg, lfsr)
    while reg != seed && n < (1 << lfsr.k)
        step!(reg, lfsr)
        n += 1
    end
    return n
end

lfsr = LFSR(9, 5, 16, 1)                 # primitive polynomial z^9 + z^4 + 1
reg = UInt8[1, 0, 0, 0, 0, 0, 0, 0, 0]   # nonzero seed

println("First 10 outputs:")
for _ in 1:10
    println(next_u!(reg, lfsr))
end

println("\nPeriod of the bit sequence = ", period(lfsr, reg),
        " (expected 2^9-1 = ", (1 << 9) - 1, ")")

# the bit-state period is always 2^k - 1; the period of the OUTPUT sequence
# (word n reads bit n*nu) is (2^k-1)/gcd(nu, 2^k-1), hence we need gcd(nu,2^k-1)=1
function output_period(lfsr::LFSR, seed::Vector{UInt8})
    reg = copy(seed)
    u0 = next_u!(reg, lfsr)
    n = 1
    while n < (1 << lfsr.k)
        u = next_u!(reg, lfsr)
        u == u0 && return n
        n += 1
    end
    return -1
end
println("\nEffect of nu on the OUTPUT period (2^9-1 = 511):")
for nu in 1:7
    l = LFSR(9, 5, 16, nu)
    p = output_period(l, UInt8[1, 0, 0, 0, 0, 0, 0, 0, 0])
    println("  nu = $nu -> output period = $p (expected ",
            p < 0 ? "n/a" : string(511 ÷ gcd(nu, 511)), ")")
end

# moment / uniformity check on 10^5 draws
reg = UInt8[1, 0, 0, 0, 0, 0, 0, 0, 0]
n = 100_000
s = sum(next_u!(reg, lfsr) for _ in 1:n) / n
println("\nMean of $n draws: ", s, " (expect ~0.5)")
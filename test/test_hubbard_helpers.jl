
"""
Helper utilities and `particle_ket` constructors for spinful fermion occupation
vectors (bitstrings).

Conventions:
 - Modes are arranged as (site0_up, site0_down, site1_up, site1_down, ...).
 - `N` is the number of modes (e.g. 8 for 4 sites × 2 spins).
 - `Sz` is the total spin-z: (N_up - N_down)/2 (can be integer or half-integer).
 - `basis` is a Vector of bit-vectors, each bit-vector is Vector{Int} with 0/1.

The `particle_ket` returns a tuple `(ket_obj, occ)` where `occ` is a Vector{Int}.
If a `Ket` type is defined in your environment (e.g. from a package), it will
attempt to construct `Ket(occ)` and return it as `ket_obj`. Otherwise `ket_obj`
is returned as `nothing`.
"""
module HubbardHelpers

export count_particles, spin_z, indices_sector, particle_ket

using Random

# --- Helpers for particle counts / Sz ---

"""
    count_particles(bits::AbstractVector{<:Integer}) -> Int

Count number of occupied modes (1s) in `bits`.
"""
count_particles(bits::AbstractVector{<:Integer}) = sum(bits)

"""
    spin_z(bits::AbstractVector{<:Integer}) -> Real

Compute total S_z = (N_up - N_down)/2 assuming up spins are in indices
1,3,5,... and down spins are in indices 2,4,6,... 
"""
function spin_z(bits::AbstractVector{<:Integer})
    @assert length(bits) % 2 == 0 "Number of modes should be even (pairs up/down per site)."
    up   = sum(bits[1:2:end])
    down = sum(bits[2:2:end])
    return (up - down) / 2
end

"""
    indices_sector(basis::Vector{Vector{Int}}; N::Union{Int,Nothing}=nothing, Sz::Union{Real,Nothing}=nothing)
Return indices (1-based) of `basis` vectors that satisfy particle number `N` and/or total `Sz`.
If `N` or `Sz` is `nothing`, that constraint is ignored.
"""
function indices_sector(basis::Vector{Vector{Int}}; N::Union{Int,Nothing}=nothing, Sz::Union{Real,Nothing}=nothing)
    inds = Int[]
    for (i, bits) in enumerate(basis)
        if (N === nothing || count_particles(bits) == N) && (Sz === nothing || spin_z(bits) == Sz)
            push!(inds, i)
        end
    end
    return inds
end

# ---------------------
# particle_ket function
# ---------------------

"""
    particle_ket(N::Int, Nparticles::Int, Sz::Real; mode=:first, basis=nothing, rng=GLOBAL_RNG)

Return `(ket_obj, occ)` where `occ` is a Vector{Int} of length `N` with exactly `Nparticles` ones
(obeying `Sz` if possible).  `ket_obj` is `Ket(occ)` if a `Ket` constructor exists, otherwise `nothing`.

Modes:
 - `:first`     -> occupy the first `Nparticles` modes (|111000...>)
 - `:random`    -> randomly choose `Nparticles` occupied modes
 - `:alternate` -> pattern 1,0,1,0,... adjusted to have `Nparticles` ones
 - `:subspace`  -> requires a `basis` argument (Vector of bit-vectors); pick a basis vector uniformly
                  from the sector with exactly `Nparticles` and `Sz` (randomly if multiple).
 - `:c4_invariant` -> special symmetry-aware construction for a 2×2 plaquette (N must be 8).
                     It builds a state whose site-occupations are identical
                     on the four sites (strictly invariant under 90° rotation).
                     This requires `Nparticles` to be divisible by 4 (integer per-site occupation).
- `:Neel`      -> Néel classical antiferromagnetic pattern at half-filling (Nparticles = N/2).
                        Alternating up/down spins on sublattices.
                        Optionally specify `flavor` keyword argument as `:Aup` (default) or `:Bup`
                        to choose which sublattice has up spins first.

"""
function particle_ket(N::Int, Nparticles::Int, Sz::Real; mode=:first, basis::Union{Vector{Vector{Int}},Nothing}=nothing, rng::AbstractRNG=Random.GLOBAL_RNG; kwargs...)
    @assert 0 ≤ Nparticles ≤ N "Number of particles must be between 0 and N"
    @assert iseven(N) "Number of modes N should be even (pairs up/down per site assumed)."

    occ = zeros(Int, N)

    if mode == :random
        idxs = randperm(rng, N)[1:Nparticles]
        occ[idxs] .= 1

    elseif mode == :alternate
        occ[1:2:N] .= 1  # 1,0,1,0,...
        # adjust to requested number of particles
        n1 = count_particles(occ)
        if n1 > Nparticles
            idxs = findall(x->x==1, occ)
            occ[idxs[(Nparticles+1):end]] .= 0
        elseif n1 < Nparticles
            idxs = findall(x->x==0, occ)
            occ[idxs[1:(Nparticles - n1)]] .= 1
        end

    elseif mode == :first
        occ[1:Nparticles] .= 1

    elseif mode == :subspace
        if basis === nothing
            error("mode=:subspace requires a `basis` argument (Vector of bit-vectors).")
        end
        # find indices in given basis that satisfy N and Sz
        sector_inds = indices_sector(basis; N=Nparticles, Sz=Sz)
        if isempty(sector_inds)
            error("No states found in provided basis with N=$(Nparticles), Sz=$(Sz).")
        end
        chosen = rand(rng, sector_inds)
        occ .= basis[chosen]  # copy bits into occ

    elseif mode == :c4_invariant
        # For 2x2 plaquette: 4 lattice sites, each site has 2 modes (up/down) => N == 8
        if N != 8
            error(":c4_invariant is only implemented for N == 8 (4 sites x 2 spins).")
        end
        # Per-site particle count must be integer: Nparticles / 4
        if Nparticles % 4 != 0
            error("For strict C4-invariant *product* occupancy pattern on sites, Nparticles must be divisible by 4.")
        end
        per_site = div(Nparticles, 4)  # 0,1 or 2
        # Need to choose a per-site occupancy vector of length 2 (up,down) that sums to per_site
        # Possibilities: (0,0), (1,0), (0,1), (1,1). Check Sz constraint.
        # We'll choose the first option that matches Sz, otherwise error.
        candidates = Dict(
            0 => [(0,0)],
            1 => [(1,0), (0,1)],
            2 => [(1,1)]
        )
        found = false
        for (up_bit, down_bit) in candidates[per_site]
            # compute total Sz for repeating this per-site pattern on 4 sites
            tot_Sz = 4 * ((up_bit - down_bit) / 2)
            if isapprox(tot_Sz, Sz; atol=1e-12)
                # build occ: for site i, mode index mapping (site i): up -> 2*i-1, down -> 2*i
                for site in 0:3
                    occ[2*site + 1] = up_bit
                    occ[2*site + 2] = down_bit
                end
                found = true
                break
            end
        end
        if !found
            error("No per-site uniform occupancy pattern found that yields Sz=$(Sz) with per-site particle count=$(per_site).")
        end

    elseif mode == :Neel
        # Néel classical AFM: one electron per site, alternating up/down on sublattices.
        # Require half-filling: Nparticles == number_sites (N/2)
        number_sites = div(N, 2)
        @assert Nparticles == number_sites "Neel mode expects half-filling: Nparticles must equal number of sites (N/2)."
        # default flavor: which sublattice has up spins first; :Aup places up on site0, :Bup flips
        flavor = get(kwargs, :flavor, :Aup)
        # build pattern: sites labeled 0..(number_sites-1)
        # choose sublattice according to parity of site index (0 -> A, 1 -> B)
        if flavor == :Aup
            for site in 0:(number_sites-1)
                if isodd(site) # B sublattice -> down
                    occ[2*site + 1] = 0   # up
                    occ[2*site + 2] = 1   # down
                else            # A sublattice -> up
                    occ[2*site + 1] = 1
                    occ[2*site + 2] = 0
                end
            end
        elseif flavor == :Bup
            # flip pattern
            for site in 0:(number_sites-1)
                if isodd(site) # B sublattice -> up
                    occ[2*site + 1] = 1
                    occ[2*site + 2] = 0
                else            # A sublattice -> down
                    occ[2*site + 1] = 0
                    occ[2*site + 2] = 1
                end
            end
        else
            error("Unknown Neel flavor: $flavor. Use :Aup or :Bup.")
        end


    else
        error("Unknown mode: $mode")
    end

    # sanity check
    @assert count_particles(occ) == Nparticles "Constructed occupation vector has wrong particle number."

    # Ensure Sz matches (if not nothing)
    if !isnothing(Sz)
        @assert isapprox(spin_z(occ), Sz; atol=1e-12) "Constructed occupation vector has Sz=$(spin_z(occ)) but requested Sz=$(Sz)."
    end

    # Try to create a Ket if available; otherwise return nothing for ket_obj.
    ket_obj = nothing
    try
        if @isdefined(Ket)
            ket_obj = Ket(occ)   # user environment may provide Ket constructor
        end
    catch err
        # ignore errors constructing Ket, return nothing in ket_obj
        ket_obj = nothing
    end

    return ket_obj, occ
end

end # module HubbardHelpers

using .HubbardHelpers
using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2


# create a trivial basis for N=8 (all 256 bitstrings)
basis = [collect(digits(i, base=2, pad=8)) for i in 0:255]      # digits gives least-significant-first
basis = [reverse(b) for b in basis]  # now bit 1 corresponds to leftmost

# pick a random half-filled Sz=0 state from the basis
ket, occ = HubbardHelpers.particle_ket(8, 4, 0.0; mode=:subspace, basis=basis, rng=MersenneTwister(1234))
display(ket)

# make a c4-invariant state with Nparticles = 4 and Sz = 0 (each site singly occupied with a singlet-like pattern cannot be
# represented as a single occupancy vector, but per-site uniform occupancy with per_site=1 and Sz=0 corresponds to
# alternating up/down per site for per-site pattern (0,1) or (1,0); here we need to pick one candidate.)
ket2, occ2 = HubbardHelpers.particle_ket(8, 4, 0.0; mode=:c4_invariant)
display(ket)

# Get The Neel state at half-filling (N=8, Nparticles=4, Sz=0) (standard 2D Lattice)
ket, occ = HubbardHelpers.particle_ket(8, 4, 0.0; mode=:Neel, flavor=:Aup)
display(ket)
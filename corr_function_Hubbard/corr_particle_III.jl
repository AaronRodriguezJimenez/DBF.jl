#"""
#Improved particle-particle correlation function for the Hubbard model.
#"""

using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2
using Plots

# ----------------- LATTICE LAYOUT HELPERS -----------------

"""
    build_site_index_map(Lx, Ly; layout=:standard)

Return a vector `site_index` of length Nsites such that `site_index[k]` is the site number
assigned to the lattice cell with linear index k (k from 1..Nsites) under the requested layout.

Layouts supported:
 - :standard -> row-major numbering left->right each row
 - :zigzag    -> row-major but alternate rows reversed in numbering (zigzag pattern)
 
The returned mapping is such that coordinates (row, col) map to a site number.
"""
function build_site_index_map(Lx::Int, Ly::Int; layout::Symbol=:standard)
    # We'll build an Lx by Ly grid (Lx columns, Ly rows).
    # Convention: rows y = 1..Ly (top to bottom), cols x = 1..Lx (left to right).
    Nsites = Lx * Ly
    site_index = fill(0, Nsites)   # site_index[pos] => site number
    pos = 1
    for row in 1:Ly
        # compute ordering of cols depending on layout and row parity (for zigzag)
        cols = 1:Lx
        if layout == :zigzag && isodd(row) == false
            # for zigzag layout, flip every second row (row=2,4,...)
            cols = reverse(cols)
        end
        for col in cols
            # site number assignment: we want site numbers starting at 1 across the full grid
            # standard numbering: site_num = (row-1)*Lx + col
            site_num = (row-1)*Lx + col
            site_index[pos] = site_num
            pos += 1
        end
    end
    return site_index
end

"""
    site_to_orbitals_from_map(site_num, Nsites)

Return the (orb_up, orb_down) orbital indices for a physical site number.
standard mapping: up_orb = 2*(site_num-1)+1, down_orb = up_orb+1
"""
site_to_orbitals(site_num::Int) = (2*(site_num-1) + 1, 2*(site_num-1) + 2)

# ----------------- OCCUPATION BUILDERS -----------------

"""
    build_occ_from_site_pattern(Lx, Ly; layout=:standard, pattern=:Neel, custom_sites=nothing)

Build an `occ` vector of length N = 2*Nsites (one entry per orbital). 1 means occupied.
Supported patterns:
 - :Neel    -> half-filling "Néel" pattern: A sublattice gets ↑ occupied, B gets ↓ occupied
               (so exactly one electron per site, but spin alternates)
 - :full_up -> all up orbitals occupied
 - :full_down -> all down orbitals occupied
 - :empty   -> no occupied orbitals
 - :custom  -> user provides `custom_sites`, which can be:
      * Vector{Int} of site numbers -> occupy both spins on those sites
      * Vector{Tuple{Int,Symbol}} e.g. [(3,:up), (5,:down), (2,:both)] to occupy specific spins
"""
function build_occ_from_site_pattern(Lx::Int, Ly::Int; layout::Symbol=:standard,
                                     pattern::Symbol=:Neel, custom_sites=nothing)
    Nsites = Lx * Ly
    N = 2 * Nsites
    occ = zeros(Int, N)   # occupancy per orbital

    # site index map (position ordering not strictly necessary here, but provided for clarity)
    # For site-based patterns we just use site numbers 1..Nsites
    # The layout only affects how you interpret printed/site-number ordering elsewhere.
    site_map = build_site_index_map(Lx, Ly; layout=layout)

    if pattern == :Neel
        # assign A/B sublattice by (x+y) parity.
        # For this we need x,y from site number: site_num = (row-1)*Lx + col
        for s in 1:Nsites
            row = fld((s-1), Lx) + 1
            col = ((s-1) % Lx) + 1
            # A-sublattice if (row+col) is even (choose convention)
            if iseven(row + col)
                # occupy up orbital on A
                up, down = site_to_orbitals(s)
                occ[up] = 1
            else
                # occupy down orbital on B
                up, down = site_to_orbitals(s)
                occ[down] = 1
            end
        end

    elseif pattern == :full_up
        for s in 1:Nsites
            up, _ = site_to_orbitals(s)
            occ[up] = 1
        end

    elseif pattern == :full_down
        for s in 1:Nsites
            _, down = site_to_orbitals(s)
            occ[down] = 1
        end

    elseif pattern == :empty
        # nothing to do; occ is all zeros

    elseif pattern == :custom
        if custom_sites === nothing
            error("custom pattern requested but custom_sites == nothing")
        end
        # If custom_sites is vector of Ints -> occupy both spins on those sites
        if all(x -> isa(x,Int), custom_sites)
            for s in custom_sites
                up, down = site_to_orbitals(s)
                occ[up] = 1
                occ[down] = 1
            end
        else
            # expect vector of tuples (site, :up/:down/:both)
            for item in custom_sites
                if isa(item,Tuple) && length(item) == 2
                    s, which = item
                    up, down = site_to_orbitals(s)
                    if which == :up
                        occ[up] = 1
                    elseif which == :down
                        occ[down] = 1
                    elseif which == :both
                        occ[up] = 1
                        occ[down] = 1
                    else
                        error("custom tuple second entry must be :up, :down, or :both")
                    end
                else
                    error("custom_sites must be Vector{Int} or Vector{Tuple{Int,Symbol}}")
                end
            end
        end

    else
        error("Unknown pattern: $pattern")
    end

    return occ
end

# ----------------- PAULI / NUMBER OPERATORS (as before) -----------------

# single-orbital number operator (1 - Z)/2
function n_op_orbital(N, orb_idx)
    O = PauliSum(N, ComplexF64)
    O += Pauli(N)
    O -= Pauli(N, Z=[orb_idx])
    O *= 0.5
    return O
end

# site operator = n_up + n_down (site number is physical site label)
function n_op_site(N, site::Int)
    up, down = site_to_orbitals(site)
    return n_op_orbital(N, up) + n_op_orbital(N, down)
end

 #site-site operator
function n_i_n_j_op(N, site_i::Int, site_j::Int)
    return n_op_site(N, site_i) * n_op_site(N, site_j)
end


# ----------------- expectation helpers (unchanged) -----------------

function get_state_sequence(N,occ)
    g = Vector{PauliBasis{N}}([])
    a = Vector{Float64}([])
    for i in 1:N
        if occ[i] == 1
            push!(g, PauliBasis(Pauli(N, X=[i])))
            push!(a, π)
        end
    end
    return g, a 
end

function get_expectation_value(O, ψ, g, θ, ϵcoeff, ϵweight)
    accum_error = 0.0
    val = zeros(length(g)) 
    err = zeros(length(g)) 
    idx = 1
    ev2 = 0.0
    for (g_i,a) in zip(g,θ)
        DBF.evolve!(O, g_i, a)
        ev1 = expectation_value(O,ψ)
        DBF.coeff_clip!(O, thresh=ϵcoeff)
        DBF.weight_clip!(O, ϵweight)
        ev2 = expectation_value(O,ψ)
        accum_error += ev2-ev1
        val[idx] = ev2
        err[idx] = accum_error
        idx += 1
    end
    return val, err, O
end

# wrappers to compute orbital/site expectation values
function expect_n_orbital(ψ::Ket{N}, occ, orb_idx::Int, g, θ, ϵcoeff, ϵweight) where N
    O = n_op_orbital(N, orb_idx)
    gref, aref = get_state_sequence(N, occ)
    for (gi, ai) in zip(gref, aref)
        DBF.evolve!(O, gi, ai)
    end
    return get_expectation_value(O, ψ, g, θ, ϵcoeff, ϵweight)
end

function expect_n_site_pair(ψ::Ket{N}, occ, site_i::Int, site_j::Int, g, θ, ϵcoeff, ϵweight) where N
    O = n_i_n_j_op(N, site_i, site_j)
    gref, aref = get_state_sequence(N, occ)
    for (gi, ai) in zip(gref, aref)
        DBF.evolve!(O, gi, ai)
    end
    return get_expectation_value(O, ψ, g, θ, ϵcoeff, ϵweight)
end

function expect_n_orbital_pair(ψ::Ket{N}, occ, orb_i::Int, orb_j::Int, g, θ, ϵcoeff, ϵweight) where N
    O = n_op_orbital(N, orb_i) * n_op_orbital(N, orb_j)
    gref, aref = get_state_sequence(N, occ)
    for (gi, ai) in zip(gref, aref)
        DBF.evolve!(O, gi, ai)
    end
    return get_expectation_value(O, ψ, g, θ, ϵcoeff, ϵweight)
end

# ----------------- MAIN: compute correlation matrices (uses chosen layout & occ) -----------------

function compute_correlations(Lx::Int, Ly::Int; layout::Symbol=:standard, pattern::Symbol=:Neel)
    Nsites = Lx * Ly
    N = 2 * Nsites

    # Build occ vector per requested layout/pattern
    occ = build_occ_from_site_pattern(Lx, Ly; layout=layout, pattern=pattern)

    # Load your saved DBF generator/angles file (adjust filename as needed)
    @load "t1e-2_hub.jld2"
    g = out["generators"]
    θ = out["angles"]

    ψ = Ket{N}(Int128(0))

    max_weight = N
    thresh = 1e-2

    # compute per-orbital averages first
    n_orb_avg = zeros(N)
    for p in 1:N
        vs, vse, _ = expect_n_orbital(ψ, occ, p, g, θ, thresh, max_weight)
        n_orb_avg[p] = vs[end]
    end

    # orbital-orbital connected correlator (N x N)
    orb_conn = zeros(N, N)
    println("Computing orbital-orbital correlators...")
    for p in 1:N, q in 1:N
        vs, _, _ = expect_n_orbital_pair(ψ, occ, p, q, g, θ, thresh, max_weight)
        nij = vs[end]
        orb_conn[p, q] = nij - n_orb_avg[p]*n_orb_avg[q]
    end

    # site-site (total) connected correlator (Nsites x Nsites) via site operators
    site_conn = zeros(Nsites, Nsites)
    n_site_avg = zeros(Nsites)
    for s in 1:Nsites
        up, down = site_to_orbitals(s)
        n_site_avg[s] = n_orb_avg[up] + n_orb_avg[down]
    end

    println("Computing site-site (total) correlators using site operators...")
    for i in 1:Nsites, j in 1:Nsites
        vs, _, _ = expect_n_site_pair(ψ, occ, i, j, g, θ, thresh, max_weight)
        nij = vs[end]
        site_conn[i, j] = nij - n_site_avg[i]*n_site_avg[j]
    end

    # spin-resolved sector matrices (↑↑, ↑↓, ↓↑, ↓↓) (site-level)
    upup = zeros(Nsites, Nsites)
    updown = zeros(Nsites, Nsites)
    downup = zeros(Nsites, Nsites)
    downdown = zeros(Nsites, Nsites)

    println("Computing spin-resolved site-sector correlators (this does 4*Nsites^2 orbital pairs)...")
    for i in 1:Nsites, j in 1:Nsites
        up_i, down_i = site_to_orbitals(i)
        up_j, down_j = site_to_orbitals(j)

        vs, _, _ = expect_n_orbital_pair(ψ, occ, up_i, up_j, g, θ, thresh, max_weight); upup[i,j]    = vs[end] - n_orb_avg[up_i]*n_orb_avg[up_j]
        vs, _, _ = expect_n_orbital_pair(ψ, occ, up_i, down_j, g, θ, thresh, max_weight); updown[i,j]  = vs[end] - n_orb_avg[up_i]*n_orb_avg[down_j]
        vs, _, _ = expect_n_orbital_pair(ψ, occ, down_i, up_j, g, θ, thresh, max_weight); downup[i,j]  = vs[end] - n_orb_avg[down_i]*n_orb_avg[up_j]
        vs, _, _ = expect_n_orbital_pair(ψ, occ, down_i, down_j, g, θ, thresh, max_weight); downdown[i,j] = vs[end] - n_orb_avg[down_i]*n_orb_avg[down_j]
    end

    # sanity check: site_conn == sum of spin sectors
    diff_check = maximum(abs.(site_conn - (upup + updown + downup + downdown)))
    @printf("Max abs difference between site_conn and sum(spin sectors) = %e\n", diff_check)

    return Dict(
        :layout => layout,
        :pattern => pattern,
        :occ => occ,
        :n_orb_avg => n_orb_avg,
        :orb_conn => orb_conn,
        :n_site_avg => n_site_avg,
        :site_conn => site_conn,
        :spin_sectors => (upup=upup, updown=updown, downup=downup, downdown=downdown),
        :site_map => build_site_index_map(Lx, Ly; layout=layout),
    )
end

# ---------- run and build matrices ----------
function run()
    # lattice parameters
    Lx = 10
    Ly = 1
    Nsites = Lx * Ly

    # choose layout and pattern
    layout = :zigzag    # or :standard
    pattern = :Neel

    println("Computing correlations for Lx=$Lx Ly=$Ly layout=$layout pattern=$pattern ...")
    res = compute_correlations(Lx, Ly; layout=layout, pattern=pattern)

    # extract results
    site_conn = res[:site_conn]           # Nsites x Nsites connected correlator
    n_site_avg = res[:n_site_avg]         # per-site averages
    site_map = res[:site_map]             # mapping (if present)
    orb_conn = res[:orb_conn]             # orbital-orbital connected correlator

    #Print results
    @printf("Per-site average occupations (total):\n")
    for s in 1:Nsites
        @printf(" site %2d: %8.4f\n", s, n_site_avg[s])
    end
    println()
    @printf("Site-site connected correlation matrix ⟨n(i) n(j)⟩_c:\n")
    println(site_conn)
    println() 
    println("Site map (physical site numbers): ", repr(site_map))
    println()
    #println("Spin sectors: ", repr(spin_sectors))

    # --------- helper: build the plotting order (site_order) ----------
    # site_order is a vector of site numbers ordered by physical grid position
    function build_site_order(Lx, Ly; layout::Symbol=:zigzag)
        site_order = Int[]
        for row in 1:Ly
            cols = 1:Lx
            if layout == :zigzag && iseven(row)
                cols = reverse(cols)
            end
            for col in cols
                push!(site_order, (row-1)*Lx + col)
            end
        end
        return site_order
    end

    site_order = build_site_order(Lx, Ly; layout=layout)
    @printf("site_order (plot order) = %s\n", repr(site_order))

    # reorder site_conn so rows/cols follow physical layout
    ordered_conn = site_conn[site_order, site_order]
    #ordered_conn = orb_conn[site_order, site_order]

    # --------- Plot 1: site-site connected correlator heatmap (ordered by layout) ----------
    xticks_pos = collect(1:length(site_order))
    yticks_pos = collect(1:length(site_order))
    xticks_labels = string.(site_order)
    yticks_labels = string.(site_order)

    heatmap(
        ordered_conn,
        xlabel = "Site j",
        ylabel = "Site i",
        title = "⟨C†_i C_j⟩",
        xticks = (xticks_pos, xticks_labels),
        yticks = (yticks_pos, yticks_labels),
        #aspect_ratio = 1,
        colorbar = true,
    )
    savefig("site_conn_ordered_heatmap.png")
    #savefig("orb_conn_ordered_heatmap.png")
    println("Saved site_conn_ordered_heatmap.png")

    # --------- Plot 2: correlation function for a chosen reference site ----------
    # Choose reference site (physical site number 1..Nsites). Use center if possible.
    ref_sites = [1, 2, 3, 4, 5]
    println("Reference sites chosen: ", ref_sites)

    for ref_site in ref_sites
        # raw correlator vs target site index (1..Nsites)
        corr_vs_site = site_conn[ref_site, :]
        # plot vs site index (physical site numbering)
        plot(1:Nsites, corr_vs_site,
            label = "connected ⟨n($ref_site) n(j)⟩_c",
            xlabel = "site j",
            ylabel = "connected correlation",
            title = "Connected correlation from site $ref_site",
        )
        savefig("corr_vs_site_ref$(ref_site).png")
        println("Saved corr_vs_site_ref$(ref_site).png")
    end

end

out = run()

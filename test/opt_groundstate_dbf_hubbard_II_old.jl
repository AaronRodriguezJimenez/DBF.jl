using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using Test
using Plots
using Dates
using Printf
using JLD2


"""
    particle_ket(N::Int, Nparticles::Int; mode=:first)

Return a `Ket` of length `N` with exactly `Nparticles` ones.

`mode` options:
  • `:first`   → first `Nparticles` sites occupied (|111000...>)
  • `:random`  → random positions for the 1s
  • `:alternate` → alternating pattern |101010...> (if Nparticles ≈ N/2)

If the alternating pattern has more or fewer 1s than `Nparticles`,
the result is truncated or padded with zeros.
"""
function particle_ket(N::Int, Nparticles::Int; mode=:first)
    @assert 0 ≤ Nparticles ≤ N "Number of particles must be between 0 and N"
    occ = zeros(Int, N)

    if mode == :random
        occ[randperm(N)[1:Nparticles]] .= 1

    elseif mode == :alternate
        occ[1:2:N] .= 1  # pattern 1,0,1,0,...
        # adjust to requested number of particles
        n1 = count(==(1), occ)
        if n1 > Nparticles
            # trim some 1s from the end
            idxs = findall(==(1), occ)
            occ[idxs[(Nparticles+1):end]] .= 0
        elseif n1 < Nparticles
            # add 1s where there are zeros
            idxs = findall(==(0), occ)
            occ[idxs[1:(Nparticles - n1)]] .= 1
        end

    else  # :first
        occ[1:Nparticles] .= 1
    end

    return Ket(occ)
end

# --- saver function (fixed escaping & robust) ---
function save_dbf_results(; N, threshold, wmax, wtype, dbfEs, nterms, loss,
                           outdir="./results", prefix="DBF")
    mkpath(outdir)

    sanitize_for_filename(s) = replace(string(s), r"[\/\:\*\?\"<>\|\s]" => "_")

    fname_w = (wmax === nothing) ? "All" : string(wmax)
    wname   = (wtype == 0) ? "Pauli" : "Majorana"

    baseprefix = joinpath(outdir,
        "$(prefix)_N=$(sanitize_for_filename(N))_th=$(sanitize_for_filename(threshold))_w=$(sanitize_for_filename(fname_w))_type=$(sanitize_for_filename(wname))"
    )


    timestamp = replace(string(Dates.now()), r"[:\s]" => "-")
    hdr = join([
        "# DBF results",
        "# N = $(N)",
        "# threshold = $(threshold)",
        "# wmax = $(fname_w)",
        "# wtype = $(wtype)  # ($(wname))",
        "# Generated on $(timestamp)",
        "# Columns: step <tab> value",
        ""
    ], "\n")

    function write_column_file(filepath, colname, arr)
        try
            open(filepath, "w") do io
                write(io, hdr, "\n")
                write(io, "# $(colname)\n")
                for i in 1:length(arr)
                    write(io, string(i, '\t', arr[i], '\n'))
                end
            end
            println("Wrote: $filepath")
            return filepath
        catch e
            println("Failed to write $filepath : ", e)
            try
                partial = filepath * ".partial"
                open(partial, "w") do io
                    write(io, "# Partial dump after error: $(e)\n")
                    write(io, hdr, "\n")
                    write(io, "# $(colname) (partial)\n")
                    for i in 1:length(arr)
                        write(io, string(i, '\t', arr[i], '\n'))
                    end
                end
                println("Wrote partial dump to: $partial")
                return partial
            catch e2
                println("Also failed to write partial: ", e2)
                return nothing
            end
        end
    end

    energy_file = baseprefix * "_Energies.txt"
    nterms_file = baseprefix * "_N-terms.txt"
    loss_file   = baseprefix * "_Loss.txt"
    all_file    = baseprefix * "_AllData.txt"

    p_energy = write_column_file(energy_file, "Energies (a.u.)", dbfEs)
    p_nterms = write_column_file(nterms_file, "Number of Hamiltonian terms", nterms)
    p_loss   = write_column_file(loss_file, "Loss (1 - HS-norm^2)", loss)

    return Dict(
        :energies => p_energy,
        :nterms   => p_nterms,
        :loss     => p_loss,
        :outdir   => outdir,
    )
end

# --- corrected run function ---
function run(; U=U, threshold=1e-3, wmax=nothing, wtype=0, read_from_file=false)
    # Working directories:
    workdir = "/Users/admin/VSCProjects/DBF.jl/"
    hamiltonian_filename = "H_evolved_U_$(U)_threshold_$(threshold)_wmax_$(wmax)_wtype_$(wtype).jld2"
    workfile = joinpath(workdir, hamiltonian_filename)

    # Parameters for Hubbard model
    Lx = 2
    Ly = 2
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    t = 0.1

    if read_from_file && isfile(workfile)
        println("Loading Hamiltonian from file: ", workfile)
        @load workfile H
    else
        println("Computing Hamiltonian")
        H = DBF.fermi_hubbard_2D(Lx, Ly, t, U)
    end

    println("Initial Hamiltonian:")
    display(H)

    # Initial state: half-filling
    Nparticles = Nsites ÷ 2
    #ψ = particle_ket(N, Nparticles, mode=:random)
    ψ = particle_ket(N, Nparticles, mode=:first)
    #ψ = particle_ket(N, Nparticles, mode=:alternate)
    display(ψ)

    e0 = expectation_value(H, ψ)
    @printf(" E0 = %12.8f\n", e0)

    H, dbfEs, nterms, loss = dbf_groundstate_test(H, ψ, max_iter=200, conv_thresh=1e-6,
                                evolve_coeff_thresh=threshold,
                                evolve_weight_thresh=wmax, w_type=wtype,
                                search_n_top=10000)

    println(" New H:")
    @save workfile H
    
    # Save results to text files (customize outdir if desired)
    out = save_dbf_results(
        N = N, threshold = threshold, wmax = wmax, wtype = wtype,
        dbfEs = dbfEs, nterms = nterms, loss = loss,
        outdir = workdir
    )

    println("Saved DBF outputs: ", out)

    e1 = expectation_value(H, ψ)
    @printf(" E1 = %12.8f\n", e1)

    variance = DBF.variance(H, ψ)
    return variance, dbfEs, nterms, loss
end



#= 
   Test set performing comparisons of the quality of the DBF-OPT ground state solver for the Hubbard model.
   Here we forculs solely on the 2D Hubbard model on a 2x2 lattice. which can be solved exactly.
   
   - Error comparison and performance with comparison to exact diagonalization
   - Comparison of different weight and coeff thresholding pruning strategies
   - How accurate is this approach with differen choices of the parameters?

   Comparisons in different coupling regimes:
      Weak coupling regime: t=0.1, U=0.001
      Middle coupling regime: t=0.1, U=0.09
      Strong coupling regime: t=0.1, U=0.5   
=#
us = [0.5]
threshs = [1e-2, 1e-3]
Pweights = [2, 3, 4, 5, 6, 7, 8]
Mweights = [2, 3, 4, 5, 6, 7, 8]

# Persistent accumulators for comparison plots (label -> variance)
Pauli_variances   = Dict{String,Float64}()
Majorana_variances = Dict{String,Float64}()

read_from_file = false

# Helper: produce a safe scientific-string for filenames
safe_sci(x) = @sprintf("%.0e", x)    # e.g. 1e-02

# Helper: parse "th=..., w=..." label -> (th, w)
function parse_th_w(label::String)
    m = match(r"th=([0-9.eE+-]+),\s*w=([0-9]+)", label)
    if m !== nothing
        th = parse(Float64, m.captures[1])
        w  = parse(Int,    m.captures[2])
        return (th, w)
    else
        return (Inf, Inf)
    end
end

for U in us
    println("========================================")
    println(" U = ", U)
    println("========================================")
    println("---- Coefficient Thresholding Only ----")

    # Per-experiment lists (typed)
    variance_list = Float64[]                 # scalar per threshold
    dbfEs_list    = Vector{Float64}[]         # list-of-vectors
    nterms_list   = Vector{Int}[]             # list-of-vectors (or could be scalars)
    loss_list     = Vector{Float64}[]         # list-of-vectors
    labels_list   = String[]                  # one label per threshold

    # --- Coefficient-only sweep ---
    for thresh in threshs
        var, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=nothing, wtype=0, read_from_file=read_from_file)

        # Print debugging info (useful to confirm shapes)
        println("th = ", thresh)
        println(" dbfEs type: ", typeof(dbfEs), " length: ", (isa(dbfEs,AbstractArray) ? length(dbfEs) : 1))
        println(" nterms type: ", typeof(nterms), " length: ", (isa(nterms,AbstractArray) ? length(nterms) : 1))
        println(" loss  type: ", typeof(loss), " length: ", (isa(loss,AbstractArray) ? length(loss) : 1))
        println(" var   type: ", typeof(var))

        push!(variance_list, var)
        push!(dbfEs_list, dbfEs)
        push!(nterms_list, isa(nterms,AbstractArray) ? nterms : [nterms])  # normalize to vector
        push!(loss_list,    isa(loss,AbstractArray)  ? loss  : [loss])
        push!(labels_list, @sprintf("th=%.0e", thresh))

        @printf(" Variance with coeff thresholding only (th=%1.1e): %1.5e\n", thresh, var)
    end

    # Overlay plots for coefficient-only
    pltE = plot()
    pltN = plot()
    pltL = plot()
    for i in 1:length(dbfEs_list)
        dbfEs = dbfEs_list[i]
        if !(isa(dbfEs, AbstractVector) && length(dbfEs) > 0)
            @warn "dbfEs for index $i is not a non-empty vector; skipping"
            continue
        end
        steps = 1:length(dbfEs)
        label = labels_list[i]

        plot!(pltE, steps, dbfEs, lw=2, label=label)
        # nterms_list[i] is normalized to vector
        if length(nterms_list[i]) == length(steps)
            plot!(pltN, steps, nterms_list[i], lw=2, label=label)
        else
            plot!(pltN, steps, fill(nterms_list[i][1], length(steps)), lw=2, label=label)
        end

        if length(loss_list[i]) == length(steps)
            plot!(pltL, steps, loss_list[i], lw=2, label=label)
        else
            plot!(pltL, steps, fill(loss_list[i][1], length(steps)), lw=2, label=label)
        end
    end

    xlabel!(pltE, "DBF Step"); ylabel!(pltE, "DBF Energy Estimate (a.u.)")
    savefig(pltE, "Energies_U=$(U)_th=varied_w=None.pdf")

    xlabel!(pltN, "DBF Step"); ylabel!(pltN, "Number of Terms")
    savefig(pltN, "Nterms_U=$(U)_th=varied_w=None.pdf")

    xlabel!(pltL, "DBF Step"); ylabel!(pltL, "Loss (1 - HS-norm^2)")
    savefig(pltL, "Loss_U=$(U)_th=varied_w=None.pdf")

    # a simple variance vs experiment plot
    pltVar = plot(1:length(variance_list), variance_list, lw=2, marker=:circle, label="Variance")
    xlabel!(pltVar, "Experiment index"); ylabel!(pltVar, "Variance")
    savefig(pltVar, "Variance_U=$(U)_th=varied_w=None.pdf")

    # --- Coefficient + Pauli weight sweep (make one overlay per threshold) ---
    println("---- Coefficient + Pauli Weight Thresholding ----")

    for thresh in threshs
        # per-threshold typed containers
        variance_list_th = Float64[]
        dbfEs_list_th    = Vector{Float64}[]
        nterms_list_th   = Vector{Int}[]
        loss_list_th     = Vector{Float64}[]
        labels_list_th   = String[]

        for wmax in Pweights
            varianceP, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=wmax, wtype=0, read_from_file=read_from_file)

            push!(variance_list_th, varianceP)
            push!(dbfEs_list_th, isa(dbfEs,AbstractArray) ? dbfEs : [dbfEs])
            push!(nterms_list_th, isa(nterms,AbstractArray) ? nterms : [nterms])
            push!(loss_list_th,   isa(loss,AbstractArray)  ? loss  : [loss])
            push!(labels_list_th, @sprintf("th=%.0e, w=%d", thresh, wmax))

            lbl = @sprintf("th=%.0e, w=%d", thresh, wmax)
            Pauli_variances[lbl] = varianceP
            @printf(" Variance with coeff (th=%1.1e) + Pauli weight (w=%d): %1.5e\n", thresh, wmax, varianceP)
        end

        # overlay plots for this thresh (Pauli)
        pltE = plot()
        pltN = plot()
        pltL = plot()
        for i in 1:length(dbfEs_list_th)
            dbfEs = dbfEs_list_th[i]
            if !(isa(dbfEs, AbstractVector) && length(dbfEs) > 0)
                @warn "dbfEs for index $i is not a vector; skipping"
                continue
            end
            steps = 1:length(dbfEs)
            label = labels_list_th[i]
            plot!(pltE, steps, dbfEs, lw=2, label=label)
            nvec = nterms_list_th[i]; lvec = loss_list_th[i]
            plot!(pltN, steps, length(nvec)==length(steps) ? nvec : fill(first(nvec), length(steps)), lw=2, label=label)
            plot!(pltL, steps, length(lvec)==length(steps) ? lvec : fill(first(lvec), length(steps)), lw=2, label=label)
        end

        xlabel!(pltE, "DBF Step"); ylabel!(pltE, "DBF Energy Estimate (a.u.)")
        safe_th = safe_sci(thresh)
        savefig(pltE, "Energies_U=$(U)_th=$(safe_th)_Pauli.pdf")

        xlabel!(pltN, "DBF Step"); ylabel!(pltN, "Number of Terms")
        savefig(pltN, "Nterms_U=$(U)_th=$(safe_th)_Pauli.pdf")

        xlabel!(pltL, "DBF Step"); ylabel!(pltL, "Loss (1 - HS-norm^2)")
        savefig(pltL, "Loss_U=$(U)_th=$(safe_th)_Pauli.pdf")
    end

    # --- Coefficient + Majorana weight sweep ---
    println("---- Coefficient + Majorana Weight Thresholding ----")
    for thresh in threshs
        variance_list_th = Float64[]
        dbfEs_list_th    = Vector{Float64}[]
        nterms_list_th   = Vector{Int}[]
        loss_list_th     = Vector{Float64}[]
        labels_list_th   = String[]

        for wmax in Mweights
            varianceM, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=wmax, wtype=1, read_from_file=read_from_file)

            push!(variance_list_th, varianceM)
            push!(dbfEs_list_th, isa(dbfEs,AbstractArray) ? dbfEs : [dbfEs])
            push!(nterms_list_th, isa(nterms,AbstractArray) ? nterms : [nterms])
            push!(loss_list_th,   isa(loss,AbstractArray)  ? loss  : [loss])
            push!(labels_list_th, @sprintf("th=%.0e, w=%d", thresh, wmax))

            lbl = @sprintf("th=%.0e, w=%d", thresh, wmax)
            Majorana_variances[lbl] = varianceM
            @printf(" Variance with coeff (th=%1.1e) + Majorana weight (w=%d): %1.5e\n", thresh, wmax, varianceM)
        end

        # overlay plots for this thresh (Majorana)
        pltE = plot()
        pltN = plot()
        pltL = plot()
        for i in 1:length(dbfEs_list_th)
            dbfEs = dbfEs_list_th[i]
            if !(isa(dbfEs, AbstractVector) && length(dbfEs) > 0)
                @warn "dbfEs for index $i is not a vector; skipping"
                continue
            end
            steps = 1:length(dbfEs)
            label = labels_list_th[i]
            plot!(pltE, steps, dbfEs, lw=2, label=label)
            nvec = nterms_list_th[i]; lvec = loss_list_th[i]
            plot!(pltN, steps, length(nvec)==length(steps) ? nvec : fill(first(nvec), length(steps)), lw=2, label=label)
            plot!(pltL, steps, length(lvec)==length(steps) ? lvec : fill(first(lvec), length(steps)), lw=2, label=label)
        end

        xlabel!(pltE, "DBF Step"); ylabel!(pltE, "DBF Energy Estimate (a.u.)")
        safe_th = safe_sci(thresh)
        savefig(pltE, "Energies_U=$(U)_th=$(safe_th)_Majorana.pdf")

        xlabel!(pltN, "DBF Step"); ylabel!(pltN, "Number of Terms")
        savefig(pltN, "Nterms_U=$(U)_th=$(safe_th)_Majorana.pdf")

        xlabel!(pltL, "DBF Step"); ylabel!(pltL, "Loss (1 - HS-norm^2)")
        savefig(pltL, "Loss_U=$(U)_th=$(safe_th)_Majorana.pdf")
    end

    # --- Comparison bar chart of all stored variances ---
    all_labels = collect(union(keys(Pauli_variances), keys(Majorana_variances)))
    all_labels = sort(all_labels, by = l -> parse_th_w(l))

    pauli_vals    = [get(Pauli_variances, l, NaN) for l in all_labels]
    majorana_vals = [get(Majorana_variances, l, NaN) for l in all_labels]

    # choose visible colors and reasonable fonts for readability
    pltBar = bar(
        all_labels,
        [pauli_vals majorana_vals],
        label = ["Pauli" "Majorana"],
        bar_width = 0.6,
        legend = :topright,
        rotation = 45,
        size = (1400, 900),
        xlabel = "Threshold / Weight Combination",
        ylabel = "Variance",
        title  = "Pauli vs Majorana Variances (U=$(U))",
        color = [:white :lightgray],
        linecolor = :black,
        linewidth = 0.3,
        titlefont = 14,
        guidefont = 12,
        tickfont  = 10,
        left_margin=15*Plots.mm,
        bottom_margin=30*Plots.mm,
    )

    savefig(pltBar, "Variances_Comparison_U=$(U).pdf")
    println("Saved comparison chart to Variances_Comparison_U=$(U).pdf")

    # clear the dicts for next U
    empty!(Pauli_variances)
    empty!(Majorana_variances)
end
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
threshs = [1e-2]#, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
Pweights = [2, 3, 4, 5, 6, 7, 8]
Mweights = [2, 3, 4, 5, 6, 7, 8]

Pauli_variances = Dict{String,Float64}()
Majorana_variances = Dict{String,Float64}()

variance_list = Float64[]
variance_list_Majorana = Float64[]
dbfEs_list = Vector{Float64}[]
nterms_list = Vector{Int}[]
loss_list = Vector{Float64}[]

read_from_file = false

for U in us
    println("========================================")
    println(" U = ", U)
    println("========================================")
    println("---- Coefficient Thresholding Only ----")
    for thresh in threshs
        var, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=nothing, wtype=0, read_from_file=read_from_file)

        println("dbfEs type: ", typeof(dbfEs), " length: ", length(dbfEs))
        println(" nterms type: ", typeof(nterms), " length: ", (isa(nterms, AbstractArray) ? length(nterms) : 1))
        println(" loss  type: ", typeof(loss), " length: ", (isa(loss, AbstractArray) ? length(loss) : 1))
        println(" var   type: ", typeof(var))
    

        push!(variance_list, var)
        push!(dbfEs_list, dbfEs)
        push!(nterms_list, nterms)
        push!(loss_list, loss)
        @printf(" Variance with coeff thresholding only (th=%1.1e): %1.5e\n", thresh, var)

    end
    # Overlay plots
    plt = plot()
    plt1 = plot()
    plt2 = plot()
    plt3 = plot()
    for (i, dbfEs) in enumerate(dbfEs_list)
        steps = collect(1:length(dbfEs))
        plot!(plt, steps, dbfEs, lw=2, label="th=$(threshs[i])")#, markershape=:circle)
        plot!(plt1, steps, nterms_list[i], lw=2, label="th=$(threshs[i])")#, markershape=:circle)
        plot!(plt2, steps, loss_list[i], lw=2, label="th=$(threshs[i])")#, markershape=:circle)
    end
    steps = collect(1:length(variance_list))
    plot!(plt3, steps, variance_list, lw=2, label="Variance", color=:gray)
    
    xlabel!(plt, "DBF Step")
    ylabel!(plt, "DBF Energy Estimate (a.u.)")
    savefig(plt, "Energies_U=$(U)_th=varied_w=None.pdf")

    xlabel!(plt1, "DBF Step")
    ylabel!(plt1, "Number of Terms")
    savefig(plt1, "N-Terms_U=$(U)_th=varied_w=None.pdf")

    xlabel!(plt2, "DBF Step")
    ylabel!(plt2, "Loss (1 - HS-norm^2)")
    savefig(plt2, "Loss_U=$(U)_th=varied_w=None.pdf")

    xlabel!(plt3, "Experiment ")
    ylabel!(plt3, "Variance")
    savefig(plt3, "Variance_U=$(U)_th=varied_w=None.pdf")   

   #- Clean data lists for next round
    empty!(variance_list)
    empty!(dbfEs_list)
    empty!(nterms_list)
    empty!(loss_list)   

    println("---- Coefficient + Pauli Weight Thresholding ----")
    labels_list  = String[]

    for thresh in threshs
        for wmax in Pweights
            # run returns varianceP, dbfEs, nterms, loss, groundE
            varianceP, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=wmax, wtype=0, read_from_file=read_from_file)

            # push per-run lists (if you need them for overlay plotting)
            push!(variance_list, varianceP)
            push!(dbfEs_list, dbfEs)
            push!(nterms_list, nterms)
            push!(loss_list, loss)
            push!(labels_list, "th=$(thresh), w=$(wmax)")

            # save summary into persistent dict (label -> error)
            lbl = "th=$(thresh), w=$(wmax)"
            Pauli_variances[lbl] = varianceP
            @printf(" Variance with coeff (th=%1.1e) + Pauli weight (w=%d): %1.5e\n", thresh, wmax, varianceP)
        end
        # Overlay plots (unchanged logic, using dbfEs_list, nterms_list, loss_list)
        plt = plot() 
        plt1 = plot()
        plt2 = plot()
        for (i, dbfEs) in enumerate(dbfEs_list)
            steps = collect(1:length(dbfEs))
            lbl = labels_list[i]
            plot!(plt, steps, dbfEs, lw=2, label=lbl)
            plot!(plt1, steps, nterms_list[i], lw=2, label=lbl)
            plot!(plt2, steps, loss_list[i], lw=2, label=lbl)
        end

        xlabel!(plt, "DBF Step"); ylabel!(plt, "DBF Energy Estimate (a.u.)")
        savefig(plt, "Energies_U=$(U)_th=varied_w=Pauli.pdf")

        xlabel!(plt1, "DBF Step"); ylabel!(plt1, "Number of Terms")
        savefig(plt1, "N-Terms_U=$(U)_th=varied_w=Pauli.pdf")

        xlabel!(plt2, "DBF Step"); ylabel!(plt2, "Loss (1 - HS-norm^2)")
        savefig(plt2, "Loss_U=$(U)_th=varied_w=Pauli.pdf")

        #- Clean transient data lists for next threshold round (keeps Pauli_variances dict)
        empty!(variance_list)
        empty!(dbfEs_list)
        empty!(nterms_list)
        empty!(loss_list)
    end

    println("---- Coefficient + Majorana Weight Thresholding ----")
    empty!(labels_list)

    for thresh in threshs
        for wmax in Mweights
            varianceM, dbfEs, nterms, loss = run(U=U, threshold=thresh, wmax=wmax, wtype=1, read_from_file=read_from_file)
            push!(variance_list_Majorana, varianceM)
            push!(dbfEs_list, dbfEs)
            push!(nterms_list, nterms)
            push!(loss_list, loss)
            push!(labels_list, "th=$(thresh), w=$(wmax)")

            lbl = "th=$(thresh), w=$(wmax)"
            Majorana_variances[lbl] = varianceM

            @printf(" Variance with coeff (th=%1.1e) + Majorana weight (w=%d): %1.5e\n", thresh, wmax, varianceM)
        end

        # Overlay plots for Majorana (keep file names different to avoid overwriting)
        plt = plot()
        plt1 = plot()
        plt2 = plot()

        for (i, dbfEs) in enumerate(dbfEs_list)
            steps = collect(1:length(dbfEs))
            lbl = labels_list[i]
            # NOTE: using Mweights / threshs for labels could be more accurate here;
            # keep the pattern similar to the Pauli block.
            plot!(plt, steps, dbfEs, lw=2, label=lbl)
            plot!(plt1, steps, nterms_list[i], lw=2, label=lbl)
            plot!(plt2, steps, loss_list[i], lw=2, label=lbl)
        end

        xlabel!(plt, "DBF Step"); ylabel!(plt, "DBF Energy Estimate (a.u.)")
        savefig(plt, "Energies_U=$(U)_th=varied_w=Majorana.pdf")

        xlabel!(plt1, "DBF Step"); ylabel!(plt1, "Number of Terms")
        savefig(plt1, "N-Terms_U=$(U)_th=varied_w=Majorana.pdf")

        xlabel!(plt2, "DBF Step"); ylabel!(plt2, "Loss (1 - HS-norm^2)")
        savefig(plt2, "Loss_U=$(U)_th=varied_w=Majorana.pdf")

    #- Clean transient lists (persisted summaries remain in Majorana_variances)
        empty!(variance_list_Majorana)
        empty!(dbfEs_list)
        empty!(nterms_list)
        empty!(loss_list)
    end

    # --- Collect all unique labels ---
    all_labels = collect(union(keys(Pauli_variances), keys(Majorana_variances)))

    # --- Define a helper to extract numeric th and w from each label safely ---
    function parse_th_w(label::String)
        m = match(r"th=([0-9.eE+-]+), w=([0-9]+)", label)
        if m !== nothing
            th = parse(Float64, m.captures[1])
            w  = parse(Int,    m.captures[2])
            return (th, w)
        else
            # fallback in case the label format is unexpected
            return (Inf, Inf)
        end
    end
    # --- Sort labels numerically by threshold, then weight ---
    all_labels = sort(all_labels, by = l -> parse_th_w(l))

    # --- Now build your arrays using the new sorted order ---
    pauli_vals    = [get(Pauli_variances, l, NaN) for l in all_labels]
    majorana_vals = [get(Majorana_variances, l, NaN) for l in all_labels]

    println("All labels: ", all_labels)

    # Plot grouped bars
    bar(
    all_labels,
    [pauli_vals majorana_vals],
    label = ["Pauli" "Majorana"],
    bar_width = 0.6,
    legend = :topright,
    legendfontsize = 30,         # larger legend text
    rotation = 45,
    size = (3500, 3000),
    xlabel = "Threshold / Weight Combination",
    ylabel = "Variance",
    title  = "Pauli vs Majorana Variances (U=$(U))",
    color = [:white :lightgray], # white and light gray bars
    linecolor = :black,
    linewidth = 0.2,
    # 👇 make everything readable
    titlefont = 50,
    guidefont = 50,              # affects xlabel & ylabel fonts
    tickfont  = 30,              # affects bar tick labels
    left_margin=25*Plots.mm,
    right_margin=25*Plots.mm,
    top_margin=30*Plots.mm,
    bottom_margin=50*Plots.mm,
)

    savefig("Variances_Comparison_U=$(U).pdf")
    println("Saved comparison chart to Variances_Comparison_U=$(U).pdf")
    # Clear dicts for next U value
    empty!(Pauli_variances)
    empty!(Majorana_variances)

end
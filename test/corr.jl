using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2
using Plots

function get_expectation_value(O, ψ, g, θ, ϵcoeff, ϵweight)
    accum_error = 0.0
    val = zeros(length(g)) 
    err = zeros(length(g)) 
    idx = 1
    ev2 = 0.0
    for (g,a) in zip(g,θ)
        
        DBF.evolve!(O, g, a)
        
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

function get_spin_correlation_function(ψ::Ket{N}, i, j, g, θ, ϵcoeff, ϵweight) where N
    S = PauliSum(N, ComplexF64)
    S += Pauli(N, X=[i,j])
    S += Pauli(N, Y=[i,j])
    S += Pauli(N, Z=[i,j])
    gref,aref = DBF.get_1d_neel_state_sequence(N)
   # gref,aref = DBF.get_rvb_sequence(N)
    for (gi, ai) in zip(gref,aref)
        DBF.evolve!(S, gi, ai)
    end
    return get_expectation_value(S, ψ, g, θ, ϵcoeff, ϵweight)
end
function get_spin_correlation_function(ψ::Ket{N}, i, g, θ, ϵcoeff, ϵweight) where N
    S = PauliSum(N, ComplexF64)
    S += Pauli(N, X=[i])
    S += Pauli(N, Y=[i])
    S += Pauli(N, Z=[i])
    gref,aref = DBF.get_1d_neel_state_sequence(N)
  #  gref,aref = DBF.get_rvb_sequence(N)
    for (gi, ai) in zip(gref,aref)
        DBF.evolve!(S, gi, ai)
    end
    return get_expectation_value(S, ψ, g, θ, ϵcoeff, ϵweight)
end

function run()
    @load "t1e-2.jld2"
    println(out)
   
    g = out["generators"]
    θ = out["angles"]
    N = 100 
    ψ = Ket{N}(Int128(0))
    
    println(" Number of rotations: ", length(g))
    i = 1

    max_weight = N
    thresh = 1e-2

    println(" Get <S(i)>")
    cs = zeros(N) 
    cse = zeros(N) 
    vs, vse, S = get_spin_correlation_function(ψ, i, g, θ, thresh, max_weight)
    @printf(" S(%3i) = %12.8f %12.8f len(S) = %12i\n", i, vs[end], vse[end], length(S))
    cs[i] = vs[end] 
    cse[i] = vs[end] - vse[end]
   
    # sites = [2,3,4,5,10,100]
    sites = collect(2:N)
    for j in sites 
        vs, vse, S = get_spin_correlation_function(ψ, j, g, θ, thresh, max_weight)
        @printf(" S(%3i) = %12.8f %12.8f len(S) = %12i\n", j, vs[end], vse[end], length(S))
        cs[j] = vs[end] 
        cse[j] = vs[end] - vse[end]
    end

    println(" Now get <S(i)S(j)>")

    
    css = zeros(N) 
    csse = zeros(N)
    for j in sites 
        i != j || continue

        vs, vse, S = get_spin_correlation_function(ψ, i, j, g, θ, thresh, max_weight)
        @printf(" S(%3i, %3i) = %12.8f %12.8f len(S) = %12i\n", i, j, vs[end], vse[end], length(S))
        css[j] = vs[end] - cs[i]*cs[j] 
        csse[j] = vs[end] - vse[end] - cse[i]*cse[j]
    end

    return css, csse
end


css, css_corrected = run()

# Define the distance or site indices
sites = collect(1:length(css))

# Plot the correlation function
plot(sites, css,
    seriestype = :scatter,
    label = "⟨S(i)S(j)⟩ - ⟨S(i)⟩⟨S(j)⟩",
    xlabel = "Site j (distance from i)",
    ylabel = "Correlation",
    title = "Spin Correlation Function",
    legend = :topright,
    markersize = 1,
    lw = 2,
)

# (Optional) overlay corrected version
plot!(sites, css_corrected,
    seriestype = :line,
    label = "Corrected",
    lw=2,
)

savefig("spin_correlation.png")
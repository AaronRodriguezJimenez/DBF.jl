using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2
using Plots

# Occupation number operator
function n_op(N, site_i)
    O = PauliSum(N, ComplexF64)
    O += Pauli(N) - Pauli(N, Z=[site_i])
    O *= 0.5
    return O
end

function n_n_op(N, i, j)
    O = PauliSum(N, ComplexF64)
    O += Pauli(N)                        # 1
    O -= Pauli(N, Z=[i])                 # -Z_i
    O -= Pauli(N, Z=[j])                 # -Z_j
    O += Pauli(N, Z=[i, j])              # + Z_i Z_j
    O *= 0.25
    return O
end


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

function get_particle_correlation_function(ψ::Ket{N}, occ, i, j, g, θ, ϵcoeff, ϵweight) where N
    n =  DBF.JWmapping(N, i=i, j=i) * DBF.JWmapping(N, i=j, j=j)
    #n = n_op(N,i)*n_op(N,j) #n_n_op(N, i, j)

    gref,aref = get_state_sequence(N, occ)
    
   # gref,aref = DBF.get_rvb_sequence(N)
    for (gi, ai) in zip(gref,aref)
        DBF.evolve!(n, gi, ai)
    end
    return get_expectation_value(n, ψ, g, θ, ϵcoeff, ϵweight)
end

function get_particle_correlation_function(ψ::Ket{N}, occ, i, g, θ, ϵcoeff, ϵweight) where N
    n =  DBF.JWmapping(N, i=i, j=i)
    #n = n_op(N, i)

    gref,aref = get_state_sequence(N, occ)
   # gref,aref = DBF.get_rvb_sequence(N)
    for (gi, ai) in zip(gref,aref)
        DBF.evolve!(n, gi, ai)
    end
    return get_expectation_value(n, ψ, g, θ, ϵcoeff, ϵweight)
end


function run()
    @load "t1e-2_hub2x2.jld2"
   
    g = out["generators"]
    θ = out["angles"]
     # Parameters for Hubbard model
    Lx = 10
    Ly = 1
    Nsites = Lx * Ly
    N = 2 * Nsites   # 2 spin states per site
    Nparticles = Nsites  # Half-filling

    ψ = Ket{N}(Int128(0))
    _, occ = DBF.particle_ket(N, Nparticles, 0.0; mode=:Neel, flavor=:Aup)
    println(" Number of rotations: ", length(g))
    i = 1 # reference site
    
    max_weight = N
    thresh = 1e-2

    println(" Get <n(i)>")
    cs = zeros(N) 
    cse = zeros(N) 
    vs, vse, n = get_particle_correlation_function(ψ, occ, i, g, θ, thresh, max_weight)
    @printf(" n(%3i) = %12.8f %12.8f len(n) = %12i\n", i, vs[end], vse[end], length(n))
    cs[i] = vs[end] 
    cse[i] = vs[end] - vse[end]
   
    # sites = [2,3,4,5,10,N]
    sites = collect(2:N)
    for j in sites 
        vs, vse, n = get_particle_correlation_function(ψ, occ, j, g, θ, thresh, max_weight)
        @printf(" n(%3i) = %12.8f %12.8f len(n) = %12i\n", j, vs[end], vse[end], length(n))
        cs[j] = vs[end]
        cse[j] = vs[end] - vse[end]
    end

    println(" Now get <n(i)n(j)>")

    
    css = zeros(N) 
    csse = zeros(N)
    for j in sites 
        i != j || continue

        vs, vse, n = get_particle_correlation_function(ψ,occ, i, j, g, θ, thresh, max_weight)
        @printf(" n(%3i, %3i) = %12.8f %12.8f len(n) = %12i\n", i, j, vs[end], vse[end], length(n))
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
    label = "⟨n(i)n(j)⟩ - ⟨n(i)⟩⟨n(j)⟩",
    xlabel = "Site j (distance from i)",
    ylabel = "Correlation",
    title = "Particle Correlation Function",
    legend = :topright,
    markersize = 2,
    lw = 2,
)

# (Optional) overlay corrected version
plot!(sites, css_corrected,
    seriestype = :line,
    label = "Corrected",
)

savefig("particle_correlation_hubbard.png")

println("sites")
println(sites)
println("css")
println(css)